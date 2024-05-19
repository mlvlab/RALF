import torch
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from .custom_fast_rcnn import CustomFastRCNNOutputLayers

@ROI_HEADS_REGISTRY.register()
class CustomStandardROIHeads(StandardROIHeads):
    @configurable
    def __init__(self, **kwargs):
        cfg = kwargs.pop('cfg')
        super().__init__(**kwargs)
        out_channels = cfg.MODEL.ROI_BOX_HEAD.FC_DIM

        self.with_image_labels = cfg.WITH_IMAGE_LABELS
        self.ws_num_props = cfg.MODEL.ROI_BOX_HEAD.WS_NUM_PROPS
        self.box_predictor = CustomFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )
        self.with_triplet_hn = cfg.WITH_TRIPLET_HN
        self.with_triplet_en = cfg.WITH_TRIPLET_EN
        self.hnen_rand_n = cfg.RAL_RAND_N
        self.use_ral = cfg.USE_RAL
    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        return ret

    def forward(self, images, features, proposals, targets=None, ann_type='box'):
        if self.training:
            del images

        features, distill_clip_features = features
        if self.training:
            if ann_type == 'box':
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

        features_box = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features_box, [x.proposal_boxes for x in proposals])
        

        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)

        if self.training and distill_clip_features is not None:
            # distilling image embedding
            distil_regions, distill_clip_embeds = distill_clip_features
            region_level_features = self.box_pooler(features_box, distil_regions)
            image_embeds = self.box_head(region_level_features)
            # image distillation
            proj_image_embeds = self.box_predictor.cls_score.linear(image_embeds)
            norm_image_embeds = F.normalize(proj_image_embeds, p=2, dim=1)
            normalized_clip_embeds = F.normalize(distill_clip_embeds, p=2, dim=1)
            distill_features = (norm_image_embeds, normalized_clip_embeds)
        else:
            distill_features = None

        if self.training and self.use_ral and ann_type == 'box':
            assert self.with_triplet_hn or self.with_triplet_en, 'Do I calculate hnen triplet loss?'

            boxes_for_hnen = [x.gt_boxes for x in targets]
            gt_classes = torch.cat([x.gt_classes for x in targets])

            assert torch.sum(gt_classes < 0) == 0
            assert torch.sum(gt_classes > 1202) == 0
            if gt_classes.shape[0] == 0:
                value_hn_en = None
            else:
                hnen_box_features = self.box_pooler(features_box, boxes_for_hnen)
                hnen_box_embeds = self.box_head(hnen_box_features)
                proj_hnen_box_embeds = self.box_predictor.cls_score.linear(hnen_box_embeds)
                norm_hnen_box_embeds = F.normalize(proj_hnen_box_embeds, p=2, dim=1)

                d, max_hn, n_cls = self.box_predictor.cls_score.hn_zs_weight.shape
                _, max_en, _ = self.box_predictor.cls_score.en_zs_weight.shape
                len_hn = self.box_predictor.cls_score.hnen_hn_pad[gt_classes]
                len_en = self.box_predictor.cls_score.hnen_en_pad[gt_classes]
                random_hn = []
                random_en = []
                for limit in len_hn:
                    random_hn.append(torch.randperm(limit, device=gt_classes.device)[:self.hnen_rand_n])
                for limit in len_en:
                    random_en.append(torch.randperm(limit, device=gt_classes.device)[:self.hnen_rand_n])
                random_hn = torch.stack(random_hn)
                random_en = torch.stack(random_en)

                b = norm_hnen_box_embeds.shape[0]

                if self.with_triplet_hn:
                    anchor_similarity = (norm_hnen_box_embeds @ self.box_predictor.cls_score.inner_zs_weight)
                    anchor = anchor_similarity[torch.arange(b), gt_classes]
                else:
                    anchor = None


                txt_feat_out_sim = self.box_predictor.cls_score.hn_zs_weight[:, :, gt_classes]
                sim_similarity_ = (norm_hnen_box_embeds @ txt_feat_out_sim.view(d, max_hn * b)).view(b, max_hn, b).permute(0, 2, 1)
                sim_ = sim_similarity_[torch.arange(b), torch.arange(b), :]
                sim_rand = sim_.gather(1, random_hn)

                if self.with_triplet_en:
                    txt_feat_out_dis = self.box_predictor.cls_score.en_zs_weight[:, :, gt_classes]
                    dissim_similarity_ = (norm_hnen_box_embeds @ txt_feat_out_dis.view(d, max_en * b)).view(b, max_en, b).permute(0, 2, 1)            
                    dissim_ = dissim_similarity_[torch.arange(b), torch.arange(b), :]
                    dissim_rand = dissim_.gather(1, random_en) 
                else:
                    dissim_rand = None

                value_hn_en = (anchor, sim_rand, dissim_rand)
        else:
            value_hn_en = None
        
        if self.training:
            del features_box
            if ann_type != 'box':
                image_labels = [x._pos_category_ids for x in targets]
                losses = self.box_predictor.image_label_losses(
                    predictions, proposals, distill_features, image_labels, value_hn_en)
                if self.use_ral:
                    assert 'loss_mask' not in losses
                    losses['loss_mask'] = predictions[0].new_zeros([1])[0]
            else:
                losses = self.box_predictor.losses(
                    (predictions[0], predictions[1]), proposals, distill_features, value_hn_en)
                # Calculate the loss for mask predictions
                losses.update(self._forward_mask(features, proposals))
                if self.with_image_labels:
                    assert 'pms_loss' not in losses
                    losses['pms_loss'] = predictions[0].new_zeros([1])[0]
            return proposals, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals, images)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def get_top_proposals(self, proposals):
        for i in range(len(proposals)):
            proposals[i].proposal_boxes.clip(proposals[i].image_size)
        proposals = [p[:self.ws_num_props] for p in proposals]
        return proposals
