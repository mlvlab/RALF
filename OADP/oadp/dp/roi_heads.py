__all__ = [
    'ViLDEnsembleRoIHead',
    'OADPRoIHead',
]

import pathlib
from typing import Any, cast

import math
import clip
import mmcv
import todd
import torch
from torch.nn import functional as F
from mmdet.core import bbox2roi, bbox2result
from mmdet.models import HEADS, BaseRoIExtractor, StandardRoIHead

from ..base import Globals
from ..base.globals_ import Store
from .bbox_heads import BlockMixin, ObjectMixin
from .cbm import CBM


@HEADS.register_module()
class ViLDEnsembleRoIHead(StandardRoIHead):
    bbox_roi_extractor: BaseRoIExtractor

    def __init__(
        self,
        *args,
        bbox_head: todd.Config,
        object_head: todd.Config,
        mask_head: todd.Config | None = None,
        **kwargs,
    ) -> None:
        # automatically detect `num_classes`
        assert bbox_head.num_classes is None
        bbox_head.num_classes = Globals.categories.num_all
        if mask_head is not None:
            assert mask_head.num_classes is None
            mask_head.num_classes = Globals.categories.num_all

        super().__init__(
            *args,
            bbox_head=bbox_head,
            mask_head=mask_head,
            **kwargs,
        )

        # `shared_head` is not supported for simplification
        assert not self.with_shared_head

        self._object_head: ObjectMixin = HEADS.build(
            object_head,
            default_args=bbox_head,
        )

        # :math:`lambda` for base and novel categories are :math:`2 / 3` and
        # :math:`1 / 3`, respectively
        lambda_ = torch.ones(Globals.categories.num_all + 1) / 3
        lambda_[:Globals.categories.num_bases] *= 2
        self.register_buffer('_lambda', lambda_, persistent=False)

    @property
    def lambda_(self) -> torch.Tensor:
        return cast(torch.Tensor, self._lambda)

    def _bbox_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Monkey patching `simple_test_bboxes`.

        Args:
            x: multilevel feature maps.
            rois: regions of interest.

        Returns:
            During training, act the same as `StandardRoIHead`.
            During Inference, replace the classification score with the
            calibrated version.

        The method breaks the `single responsibility principle`, in order for
        monkey patching `mmdet`.
        During training, the method forwards the `bbox_head` and returns the
        `bbox_results` as `StandardRoIHead` does.
        However, during inference, the method forwards both the `bbox_head`
        and the `object_head`.
        The `object_head` classifies each RoI and the predicted logits are
        used to calibrate the outputs of `bbox_head`.

        For more details, refer to ViLD_.

        .. _ViLD: https://readpaper.com/paper/3206072662
        """
        bbox_results: dict[str, torch.Tensor] = super()._bbox_forward(x, rois)
        if Globals.training:
            return bbox_results

        bbox_logits = bbox_results['cls_score']
        bbox_scores = bbox_logits.softmax(-1)**self.lambda_

        object_logits, _ = self._object_head(bbox_results['bbox_feats'])
        object_logits = cast(torch.Tensor, object_logits)
        object_scores = object_logits.softmax(-1)**(1 - self.lambda_)

        if Store.DUMP:
            self._bbox_logits = bbox_logits
            self._object_logits = object_logits

        cls_score = bbox_scores * object_scores
        cls_score[:, -1] = 1 - cls_score[:, :-1].sum(-1)

        bbox_results['cls_score'] = cls_score.log()
        return bbox_results

    def _object_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> None:
        bre = self.bbox_roi_extractor # 4 RoIAligns
        object_feats = bre(x[:bre.num_inputs], rois)
        self._object_head(object_feats) # tensor(*, 256, 7, 7)

    def object_forward_train(
        self,
        x: list[torch.Tensor],
        bboxes: list[torch.Tensor],
    ) -> None: 
        rois = bbox2roi(bboxes)
        self._object_forward(x, rois)

    if Store.DUMP:
        access_layer = todd.datasets.PthAccessLayer(
            data_root=Store.DUMP,
            readonly=False,
        )

        def simple_test_bboxes(
            self,
            x: torch.Tensor,
            img_metas: list[dict[str, Any]],
            proposals: list[torch.Tensor],
            rcnn_test_cfg: mmcv.ConfigDict,
            rescale: bool = False,
        ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
            assert x.shape[0] == len(img_metas) == len(proposals) == 1
            filename = pathlib.Path(img_metas[0]['filename']).stem
            objectness = proposals[0][:, -1]

            bboxes, _ = super().simple_test_bboxes(
                x,
                img_metas,
                proposals,
                None,
                rescale,
            )

            record = dict(
                bboxes=bboxes,
                bbox_logits=self._bbox_logits,
                object_logits=self._object_logits,
                objectness=objectness,
            )
            record = {k: v.half() for k, v in record.items()}
            self.access_layer[filename] = record

            return [torch.tensor([[0, 0, 1, 1]])], [torch.empty([0])]


@HEADS.register_module()
class OADPRoIHead(ViLDEnsembleRoIHead):

    def __init__(
        self,
        *args,
        bbox_head: todd.Config,
        block_head: todd.Config | None = None,
        test_add_score: todd.Config | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, bbox_head=bbox_head, **kwargs)
        if block_head is not None:
            self._block_head: BlockMixin = HEADS.build(
                block_head,
                default_args=bbox_head,
            )
        if test_add_score is not None:
            self._test_add_score = test_add_score
            self.clip_model, _ = clip.load_default()
            val_cls_zs_weight = self.bbox_head.fc_cls._embeddings
            val_bg_zs_weight = self.bbox_head.fc_cls._bg_embedding
            val_zs_weight = torch.cat([val_cls_zs_weight, val_bg_zs_weight])
            val_zs_weight /= val_zs_weight.norm(dim=-1, keepdim=True)
            self.val_zs_weight = val_zs_weight
            if test_add_score['is_with_cc']:
                cbm_model = CBM()
                cbm_model.load_state_dict(torch.load(test_add_score['cc_weight_path'], map_location='cpu'))
                cbm_model.eval()
                self.cbm_model = cbm_model

                concept_pkl = torch.load(test_add_score['concept_pkl_path'], map_location='cpu')
                concept_feats = concept_pkl['all_noun_chunks']['text_features']
                concept_feats = concept_feats.to(torch.float32).contiguous()
                concept_feats /= concept_feats.norm(dim=-1, keepdim=True)
                self.concept_feats = concept_feats

    @property
    def with_block(self) -> bool:
        return hasattr(self, '_block_head')

    @property
    def with_test_add_score(self) -> bool:
        return hasattr(self, '_test_add_score')

    def _block_forward(
        self,
        x: list[torch.Tensor],
        rois: torch.Tensor,
    ) -> torch.Tensor:
        bre = self.bbox_roi_extractor
        block_feats = bre(x[:bre.num_inputs], rois)
        logits, _ = self._block_head(block_feats)
        return logits

    def block_forward_train(
        self,
        x: list[torch.Tensor],
        bboxes: list[torch.Tensor],
        targets: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        rois = bbox2roi(bboxes)
        logits = self._block_forward(x, rois)
        losses = self._block_head.loss(logits[:, :-1], torch.cat(targets))
        return losses

    def _bbox_forward_gt_bboxes(self, x, gt_bboxes):
        rois = bbox2roi(gt_bboxes)
        bbox_feats = self.bbox_roi_extractor(x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:   
            bbox_feats = self.shared_head(bbox_feats)
        region_embeddings = self._bbox_head_get_bbox_feats(bbox_feats)
        return region_embeddings

    def _bbox_head_get_bbox_feats(self, x):
        '''
            bring def forward(self, x): function from mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
            x: (*, 256, 7, 7)
        '''
        if self.bbox_head.num_shared_convs > 0:
            for conv in self.bbox_head.shared_convs:
                x = conv(x)

        if self.bbox_head.num_shared_fcs > 0:
            if self.bbox_head.with_avg_pool: # False
                x = self.bbox_head.avg_pool(x)

            x = x.flatten(1) # (*, 12544)

            for fc in self.bbox_head.shared_fcs:
                x = self.bbox_head.relu(fc(x)) # fc: 12544 -> 1024
        # separate branches
        x_cls = x # (*, 1024)

        for conv in self.bbox_head.cls_convs: # False
            x_cls = conv(x_cls)
        if x_cls.dim() > 2: # False
            if self.bbox_head.with_avg_pool:
                x_cls = self.bbox_head.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.bbox_head.cls_fcs: # False
            x_cls = self.bbox_head.relu(fc(x_cls))

        region_embeddings = self.bbox_head.fc_cls._linear(x_cls)

        return region_embeddings

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False, 
                    img=None):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale, img=img)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
            return list(zip(bbox_results, segm_results))

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False, 
                           img=None):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        if self.with_test_add_score:
            assert img is not None, 'img should exist to calculate zeroshot scores.'
            zs_scores = self.get_zs_scores(img, proposals, img_shapes)
            cls_score = cls_score + zs_scores
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                det_bbox, det_label = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale,
                    cfg=rcnn_test_cfg)
            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
        return det_bboxes, det_labels

    def get_zs_scores(self, img, boxes, img_shapes):
        if self._test_add_score['is_box_level']:
            boxes = boxes[0]
            valid_mask = torch.isfinite(boxes).all(dim=1)
            if not valid_mask.all():
                boxes = boxes[valid_mask]
            num_bbox_reg_classes = boxes.shape[1] // 4
            assert num_bbox_reg_classes == 1, 'error: num_bbox_reg_classes'
            boxes = boxes[:, :4]
            img_w = img_shapes[0][1]
            img_h = img_shapes[0][0]
            boxes[:, 0::2] = torch.clamp(boxes[:, 0::2], min=0, max=img_w)
            boxes[:, 1::2] = torch.clamp(boxes[:, 1::2], min=0, max=img_h)
            boxes_int = boxes.to(torch.int32)
            
            pre_crops = []
            for box in boxes_int:
                w = box[2] - box[0]
                h = box[3] - box[1]
                assert w >= 0 and h >= 0, 'error: width or height < 0'
                assert box[2] <= img_w, 'error: width overflow'
                assert box[3] <= img_h, 'error: height overflow'
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
                crop = img[:, :, box[1]:box[3], box[0]:box[2]]
                pre_crop = self.resize_and_pad(crop)
                pre_crops.append(pre_crop)
            pre_crops = torch.stack(pre_crops)
            with torch.no_grad():
                v_f = self.clip_model.encode_image(pre_crops).to(torch.float32)
                v_f /= v_f.norm(dim=-1, keepdim=True)
            if self._test_add_score['is_with_cc']:
                with torch.no_grad():
                    gen = self.cbm_model(v_f, self.concept_feats.to(device=v_f.device))
                    sim_gen_gt = (gen @ self.val_zs_weight.to(device=gen.device).T).squeeze(0)
                    sim_v_t = sim_gen_gt
            else:
                sim_v_t = v_f @ self.val_zs_weight.to(device=v_f.device).T
            k = self._test_add_score['topk']
            ignore_idx = sim_v_t.topk(sim_v_t.shape[1] - k, largest=False).indices
            sim_v_t.scatter_(1, ignore_idx, 0, reduce='multiply')
        else:
            with torch.no_grad():
                pre_img = self.resize_and_pad(img)
                v_f = self.clip_model.encode_image(pre_img.unsqueeze(0)).to(torch.float32)
                v_f /= v_f.norm(dim=-1, keepdim=True)
            if self._test_add_score['is_with_cc']:
                with torch.no_grad():
                    gen = self.cbm_model(v_f, self.concept_feats.to(device=v_f.device))
                    sim_gen_gt = (gen @ self.val_zs_weight.to(device=gen.device).T).squeeze(0) 
                    sim_v_t = sim_gen_gt
            else:
                sim_v_t = (v_f @ self.val_zs_weight.to(device=v_f.device).T).squeeze(0)
            k = self._test_add_score['topk']
            ignore_idx = sim_v_t.topk(sim_v_t.shape[0] - k, largest=False).indices
            sim_v_t[ignore_idx] *= 0
        # return sim_v_t.sigmoid()
        return sim_v_t

    def resize_and_pad(self, img, n_px=224):
        _, _, h, w = img.shape
        max_hw = max(h, w)
        new_h = math.ceil(h / max_hw * n_px)
        new_w = math.ceil(w / max_hw * n_px)
        resized = F.interpolate(img, size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)

        padding_h = n_px - new_h
        padding_w = n_px - new_w
        top = padding_h // 2
        bottom = padding_h - top
        left = padding_w // 2
        right = padding_w - left
        padded = F.pad(resized, pad=(left, right, top, bottom), mode='constant', value=0)
        return padded
