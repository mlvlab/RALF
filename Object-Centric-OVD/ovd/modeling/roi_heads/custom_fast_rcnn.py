import math
import torch
from fvcore.nn import smooth_l1_loss
from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
import detectron2.utils.comm as comm
from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat, nonzero_tuple
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from ..utils import load_class_freq, get_fed_loss_inds
from .zero_shot_classifier import ZeroShotClassifier, WeightTransferZeroShotClassifier
from torch.nn.functional import normalize
from detectron2.structures import Boxes

__all__ = ["CustomFastRCNNOutputLayers"]


class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            cls_score=None,
            use_sigmoid_ce=False,
            use_fed_loss=False,
            ignore_zero_cats=False,
            fed_loss_num_cat=50,
            image_label_loss='',
            use_zeroshot_cls=False,
            pms_loss_weight=0.1,
            prior_prob=0.01,
            cat_freq_path='',
            fed_loss_freq_weight=0.5,
            distil_l1_loss_weight=0.5,
            irm_loss_weight=0.0,
            rkd_temperature=100,
            num_distil_prop=5,
            weight_transfer=False,
            use_ral=False,
            with_triplet_hn=False,
            with_triplet_en=False,
            lambda_hn=1.0,
            lambda_en=1.0,
            margin_hn=1.0,
            margin_en=1.0,
            strength_hn=1.0,
            strength_en=1.0,
            use_raf=False, 
            raf_topk=20,
            raf_box_level=False,
            cbm_weight='',
            **kwargs,
    ):
        super().__init__(
            input_shape=input_shape,
            **kwargs,
        )
        self.use_zeroshot_cls = use_zeroshot_cls
        self.use_sigmoid_ce = use_sigmoid_ce
        self.image_label_loss = image_label_loss
        self.pms_loss_weight = pms_loss_weight
        self.use_fed_loss = use_fed_loss
        self.ignore_zero_cats = ignore_zero_cats
        self.fed_loss_num_cat = fed_loss_num_cat
        self.distil_l1_loss_weight = distil_l1_loss_weight
        self.irm_loss_weight = irm_loss_weight
        self.rkd_temperature = rkd_temperature
        self.num_distil_prop = num_distil_prop
        self.weight_transfer = weight_transfer
        self.use_ral = use_ral
        self.with_triplet_hn = with_triplet_hn
        self.with_triplet_en = with_triplet_en
        self.lambda_hn = lambda_hn
        self.lambda_en = lambda_en
        self.margin_hn = margin_hn
        self.margin_en = margin_en
        self.strength_hn = strength_hn
        self.strength_en = strength_en
        self.use_raf = use_raf
        self.raf_topk = raf_topk
        self.raf_box_level = raf_box_level
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_score.bias, bias_value)

        if self.use_fed_loss or self.ignore_zero_cats:
            freq_weight = load_class_freq(cat_freq_path, fed_loss_freq_weight)
            self.register_buffer('freq_weight', freq_weight)
        else:
            self.freq_weight = None

        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)

        if self.use_zeroshot_cls:
            del self.cls_score
            del self.bbox_pred
            assert cls_score is not None
            self.cls_score = cls_score
            self.bbox_pred = nn.Sequential(
                nn.Linear(input_size, input_size),
                nn.ReLU(inplace=True),
                nn.Linear(input_size, 4)
            )
            weight_init.c2_xavier_fill(self.bbox_pred[0])
            nn.init.normal_(self.bbox_pred[-1].weight, std=0.001)
            nn.init.constant_(self.bbox_pred[-1].bias, 0)

        if self.use_raf:
            import clip
            self.clip_model, _ = clip.load("ViT-B/32", device="cuda")
            from ..cbm import CBM
            cbm_model = CBM()
            cbm_model.load_state_dict(torch.load(cbm_weight, map_location='cpu'))
            cbm_model.eval()
            self.cbm_model = cbm_model

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update({
            'use_zeroshot_cls': cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS,
            'use_sigmoid_ce': cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE,
            'prior_prob': cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB,
            'image_label_loss': cfg.MODEL.ROI_BOX_HEAD.IMAGE_LABEL_LOSS,
            'pms_loss_weight': cfg.MODEL.ROI_BOX_HEAD.PMS_LOSS_WEIGHT,
            'use_fed_loss': cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS,
            'ignore_zero_cats': cfg.MODEL.ROI_BOX_HEAD.IGNORE_ZERO_CATS,
            'fed_loss_num_cat': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT,
            'cat_freq_path': cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH,
            'fed_loss_freq_weight': cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
            'distil_l1_loss_weight': cfg.MODEL.DISTIL_L1_LOSS_WEIGHT,
            'irm_loss_weight': cfg.MODEL.IRM_LOSS_WEIGHT,
            'num_distil_prop': cfg.MODEL.NUM_DISTIL_PROP,
            'weight_transfer': cfg.MODEL.ROI_BOX_HEAD.WEIGHT_TRANSFER,
            'use_ral': cfg.USE_RAL,
            'with_triplet_hn': cfg.WITH_TRIPLET_HN,
            'with_triplet_en': cfg.WITH_TRIPLET_EN,
            'lambda_hn': cfg.LAMBDA_HN, 
            'lambda_en': cfg.LAMBDA_EN,
            'margin_hn': cfg.MARGIN_HN,
            'margin_en': cfg.MARGIN_EN,
            'strength_hn': cfg.STRENGTH_HN,
            'strength_en': cfg.STRENGTH_EN,
            'use_raf': cfg.USE_RAF,
            'raf_topk': cfg.RAF_TOPK,
            'raf_box_level': cfg.RAF_BOX_LEVEL,
            'cbm_weight': cfg.RAF_WEIGHT,
        })
        use_bias = cfg.MODEL.ROI_BOX_HEAD.USE_BIAS
        if ret['use_zeroshot_cls']:
            if ret['weight_transfer']:
                ret['cls_score'] = WeightTransferZeroShotClassifier(cfg, input_shape, use_bias=use_bias)
            else:
                ret['cls_score'] = ZeroShotClassifier(cfg, input_shape, use_bias=use_bias)
        return ret

    def losses(self, predictions, proposals, distil_features, value_hn_en, use_advanced_loss=True):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        num_classes = self.num_classes

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        cls_loss = self.sigmoid_cross_entropy_loss(scores, gt_classes)

        # region-based knowledge distillation loss
        if distil_features is not None:
            image_features, clip_features = distil_features
            # Point-wise embedding matching loss (L1)
            distil_l1_loss = self.distil_l1_loss(image_features, clip_features)
            if self.irm_loss_weight > 0:
                # Inter-embedding relationship matching loss (IRM)
                irm_loss = self.irm_loss(image_features, clip_features)
                return_dict = {
                    "cls_loss": cls_loss,
                    "box_reg_loss": self.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                        num_classes=num_classes),
                    "distil_l1_loss": distil_l1_loss,
                    "irm_loss": irm_loss,
                }
            else:
                return_dict = {
                    "cls_loss": cls_loss,
                    "box_reg_loss": self.box_reg_loss(
                        proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                        num_classes=num_classes),
                    "distil_l1_loss": distil_l1_loss,
                }
        else:
            return_dict = {
                "cls_loss": cls_loss,
                "box_reg_loss": self.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes,
                    num_classes=num_classes),
            }

        if value_hn_en is None:
            if self.use_ral:
                if self.with_triplet_hn:
                    return_dict.update({'hard_neg_loss': scores.new_zeros([1])[0]})
                if self.with_triplet_en:
                    return_dict.update({'easy_neg_loss': scores.new_zeros([1])[0]})
        else:
            triplet_losses = self.triplet_hnen_loss(value_hn_en)
            return_dict.update(triplet_losses)
        return return_dict

    def triplet_hnen_loss(self, value_hn_en):
        losses = {}
        anchor, v_hn, v_en = value_hn_en
        mean_hn = v_hn.mean(dim=1)

        if self.with_triplet_hn:
            hard_neg = torch.clamp(self.margin_hn + mean_hn * self.lambda_hn - anchor, 0)
            losses['hard_neg_loss'] = hard_neg.mean() * self.strength_hn

        if self.with_triplet_en:
            mean_en = v_en.mean(dim=1)
            easy_neg = torch.clamp(self.margin_en + mean_en * self.lambda_en - mean_hn, 0)
            losses['easy_neg_loss'] = easy_neg.mean() * self.strength_en
        
        return losses

    # Point-wise embedding matching loss (L1)
    def distil_l1_loss(self, image_features, clip_features):
        weight = self.distil_l1_loss_weight * self.rkd_temperature
        loss = F.l1_loss(image_features, clip_features, reduction='mean')
        loss = loss * weight
        return loss

    # Inter-embedding relationship matching loss (IRM)
    def irm_loss(self, image_features, clip_features):
        weight = self.irm_loss_weight * self.rkd_temperature
        g_img = normalize(torch.matmul(image_features, torch.t(image_features)), 1)
        g_clip = normalize(torch.matmul(clip_features, torch.t(clip_features)), 1)
        irm_loss = torch.norm(g_img - g_clip) ** 2
        return irm_loss * weight

    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0]

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1
        target = target[:, :C]

        weight = 1

        if self.use_fed_loss and (self.freq_weight is not None):
            appeared = get_fed_loss_inds(gt_classes, num_sample_cats=self.fed_loss_num_cat, C=C,
                                         weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()
        if self.ignore_zero_cats and (self.freq_weight is not None):
            w = (self.freq_weight.view(-1) > 1e-4).float()
            weight = weight * w.view(1, C).expand(B, C)

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none')
        loss = torch.sum(cls_loss * weight) / B
        return loss

    def box_reg_loss(
            self, proposal_boxes, gt_boxes, pred_deltas, gt_classes, num_classes=-1):
        """
        Allow custom background index
        """
        num_classes = num_classes if num_classes > 0 else self.num_classes
        fg_inds = nonzero_tuple((gt_classes >= 0) & (gt_classes < num_classes))[0]
        # class-agnostic regression
        fg_pred_deltas = pred_deltas[fg_inds]
        # smooth_l1 loss
        gt_pred_deltas = self.box2box_transform.get_deltas(proposal_boxes[fg_inds], gt_boxes[fg_inds], )
        box_reg_loss = smooth_l1_loss(fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum")
        return box_reg_loss / max(gt_classes.numel(), 1.0)

    def inference(self, predictions, proposals, images=None):
        """
        enable use proposal boxes
        """
        predictions = (predictions[0], predictions[1])
        boxes = self.predict_boxes(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        if self.use_raf:
            zs_probs = self.get_zeroshot_probs(images.to(boxes[0].device), boxes, image_shapes[0])
            scores = self.predict_probs(predictions, proposals, zs_probs)
        else:
            scores = self.predict_probs(predictions, proposals)
        return fast_rcnn_inference(
            boxes,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def resize_and_pad(self, img, n_px=224):
        _, h, w = img.shape
        if h == 0: h = 1
        if w == 0: w = 1
        max_hw = max(h, w)
        new_h = math.ceil(h / max_hw * n_px)
        new_w = math.ceil(w / max_hw * n_px)
        resized = F.interpolate(img.unsqueeze(0).to(torch.float32), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0).to(torch.uint8)

        padding_h = n_px - new_h
        padding_w = n_px - new_w
        top = padding_h // 2
        bottom = padding_h - top
        left = padding_w // 2
        right = padding_w - left
        padded = F.pad(resized, pad=(left, right, top, bottom), mode='constant', value=0)

        padded = padded / 255
        px_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(-1, 1, 1).to(img.device)
        px_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(-1, 1, 1).to(img.device)
        padded = (padded - px_mean) / px_std
        return padded

    def get_zeroshot_probs(self, img, boxes, image_shape):
        if self.raf_box_level:
            boxes = boxes[0]
            valid_mask = torch.isfinite(boxes).all(dim=1)
            if not valid_mask.all():
                boxes = boxes[valid_mask]
            num_bbox_reg_classes = boxes.shape[1] // 4
            assert num_bbox_reg_classes == 1, 'error: num_bbox_reg_classes'
            boxes = Boxes(boxes.reshape(-1, 4))
            boxes.clip(image_shape)
            boxes_int = boxes.tensor.to(int)
            pre_crops = []
            for box in boxes_int:
                w = box[2] - box[0]
                h = box[3] - box[1]
                assert w >= 0 and h >= 0, 'error: width or height < 0'
                assert box[2] <= image_shape[1], 'error: width overflow'
                assert box[3] <= image_shape[0], 'error: height overflow'
                if w < 1:
                    if box[0] >= 1:
                        box[0] -= 1
                    else:
                        box[2] += 1
                if h < 1:
                    if box[1] >= 1:
                        box[1] -= 1
                    else:
                        box[3] += 1
                crop = img[:, box[1]:box[3], box[0]:box[2]]
                pre_crop = self.resize_and_pad(crop)
                pre_crops.append(pre_crop)
            pre_crops = torch.stack(pre_crops)
            with torch.no_grad():
                v_f = self.clip_model.encode_image(pre_crops).to(torch.float32)
                v_f /= v_f.norm(dim=-1, keepdim=True)
                aug_v_f = self.cbm_model(v_f, self.cls_score.concept_feats)
                sim_aug_gt = (aug_v_f @ self.cls_score.zs_weight).squeeze(0)
                sim_v_t = sim_aug_gt
            k = self.raf_topk
            ignore_idx = sim_v_t.topk(sim_v_t.shape[1] - k, largest=False).indices
            sim_v_t.scatter_(1, ignore_idx, 0, reduce='multiply')
        else:
            with torch.no_grad():
                pre_img = self.resize_and_pad(img)
                v_f = self.clip_model.encode_image(pre_img.unsqueeze(0)).to(torch.float32)
                v_f /= v_f.norm(dim=-1, keepdim=True)
                aug_v_f = self.cbm_model(v_f, self.cls_score.concept_feats)
                sim_aug_gt = (aug_v_f @ self.cls_score.zs_weight).squeeze(0)
                sim_v_t = sim_aug_gt
            k = self.raf_topk
            ignore_idx = sim_v_t.topk(sim_v_t.shape[0] - k, largest=False).indices
            sim_v_t[ignore_idx] *= 0
        return sim_v_t.sigmoid()

    def predict_probs(self, predictions, proposals, zs_probs=None):
        """
        support sigmoid
        """
        scores = predictions[0]
        num_inst_per_image = [len(p) for p in proposals]
        probs = scores.sigmoid()
        if zs_probs is not None:
            probs = probs + zs_probs
        return probs.split(num_inst_per_image, dim=0)

    def image_label_losses(self, predictions, proposals, distil_features, image_labels, value_hn_en):
        """
        Inputs:
            scores: N x (C + 1)
            image_labels B x 1
        """
        num_inst_per_image = [len(p) for p in proposals]
        scores = predictions[0]
        scores = scores.split(num_inst_per_image, dim=0)
        prop_scores = [None for _ in num_inst_per_image]
        B = len(scores)
        img_box_count = 0
        select_size_count = 0
        select_x_count = 0
        select_y_count = 0
        max_score_count = 0
        storage = get_event_storage()
        loss = scores[0].new_zeros([1])[0]
        for idx, (score, labels, prop_score, p) in enumerate(zip(
                scores, image_labels, prop_scores, proposals)):
            if score.shape[0] == 0:
                loss += score.new_zeros([1])[0]
                continue
            for i_l, label in enumerate(labels):
                if self.image_label_loss == 'pseudo_max_score':
                    loss_i, ind = self._psuedo_maxscore_loss(score, label, p)
                else:
                    assert 0
                loss += loss_i / len(labels)
                if type(ind) == type([]):
                    img_box_count = sum(ind) / len(ind)
                    if self.debug:
                        for ind_i in ind:
                            p.selected[ind_i] = label
                else:
                    img_box_count = ind
                    select_size_count = p[ind].proposal_boxes.area() / (p.image_size[0] * p.image_size[1])
                    max_score_count = score[ind, label].sigmoid()
                    select_x_count = (p.proposal_boxes.tensor[ind, 0] +
                                      p.proposal_boxes.tensor[ind, 2]) / 2 / p.image_size[1]
                    select_y_count = (p.proposal_boxes.tensor[ind, 1] +
                                      p.proposal_boxes.tensor[ind, 3]) / 2 / p.image_size[0]

        loss = loss / B
        storage.put_scalar('stats_l_image', loss.item())
        if comm.is_main_process():
            storage.put_scalar('pool_stats', img_box_count)
            storage.put_scalar('stats_select_size', select_size_count)
            storage.put_scalar('stats_select_x', select_x_count)
            storage.put_scalar('stats_select_y', select_y_count)
            storage.put_scalar('stats_max_label_score', max_score_count)

        # region-based knowledge (RKD) distillation loss
        if distil_features is not None:
            image_features, clip_features = distil_features
            # Point-wise embedding matching loss (L1)
            distil_l1_loss = self.distil_l1_loss(image_features, clip_features)
            if self.irm_loss_weight > 0:
                # Inter-embedding relationship loss (IRM)
                irm_loss = self.irm_loss(image_features, clip_features)
                return_dict = {
                    'pms_loss': loss * self.pms_loss_weight,
                    'cls_loss': score.new_zeros([1])[0],
                    'box_reg_loss': score.new_zeros([1])[0],
                    'distil_l1_loss': distil_l1_loss,
                    'irm_loss': irm_loss,
                }
            else:
                return_dict = {
                    'pms_loss': loss * self.pms_loss_weight,
                    'cls_loss': score.new_zeros([1])[0],
                    'box_reg_loss': score.new_zeros([1])[0],
                    'distil_l1_loss': distil_l1_loss,
                }
        else:
            return_dict = {
                'pms_loss': loss * self.pms_loss_weight,
                'cls_loss': score.new_zeros([1])[0],
                'box_reg_loss': score.new_zeros([1])[0]
            }
        if value_hn_en is None:
            if self.use_ral:
                if self.with_triplet_hn:
                    return_dict.update({'hard_neg_loss': score.new_zeros([1])[0]})
                if self.with_triplet_en:
                    return_dict.update({'easy_neg_loss': score.new_zeros([1])[0]})
        else:
            assert True, 'This flow is not permitted.'

        return return_dict

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        scores = []
        cls_scores = self.cls_score(x)
        scores.append(cls_scores)
        scores = torch.cat(scores, dim=1)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    # pseudo-max score loss (pms loss)
    def _psuedo_maxscore_loss(self, score, label, p):
        loss = 0
        target = score.new_zeros(score.shape[1])
        target[label] = 1.
        target_keys = [p.target_proposals[l][0] for l in range(len(p.target_proposals))]
        ind = target_keys.index(label)
        # assert (p.proposal_boxes[ind].tensor.cpu().numpy() == p.target_proposals[ind][1][0]).all()
        loss += F.binary_cross_entropy_with_logits(
            score[ind], target, reduction='sum')
        return loss, ind
