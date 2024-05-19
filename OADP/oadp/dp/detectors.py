__all__ = [
    'HenHead',
    'GlobalHead',
    'OADP',
]

from typing import Any, Sequence

import einops
import todd
import json
import torch
import torch.nn.functional as F
from mmdet.models import DETECTORS, RPNHead, TwoStageDetector
from mmdet.models.utils.builder import LINEAR_LAYERS
from todd.distillers import SelfDistiller, Student
from todd.losses import LossRegistry as LR

from ..base import Globals
from .roi_heads import OADPRoIHead
from .utils import MultilabelTopKRecall


class HenHead(todd.Module):
    def __init__(self, *args, hen_path: str, random_q: int, loss_hn: todd.Config, loss_en: todd.Config, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert hen_path is not None, 'hen path is missing.'

        device = 'cpu'
        ral_info = torch.load(hen_path, map_location=device)
        anchor_weight = torch.tensor(ral_info['feat_in']).to(dtype=torch.float32)
        hard_weight = torch.tensor(ral_info['feat_out_sim']).to(dtype=torch.float32)
        easy_weight = torch.tensor(ral_info['feat_out_dis']).to(dtype=torch.float32)
        hard_pad = torch.tensor(ral_info['pad_out_sim'])
        easy_pad = torch.tensor(ral_info['pad_out_dis'])
        del ral_info

        if Globals.categories.num_all == 65: # coco
            all_json = json.load(open('data/coco/annotations/instances_val2017.json', 'rb'))
            all_cat_names_80 = [x['name'] for x in all_json['categories']]
            del all_json
            base_80_to_48 = [i for i, x in enumerate(all_cat_names_80) if x in Globals.categories.bases]
            anchor_weight = anchor_weight[base_80_to_48]

        self._hard_pad = hard_pad
        self._easy_pad = easy_pad
        self._anchor_weight = F.normalize(anchor_weight, p=2, dim=1)
        self._hard_weight = F.normalize(hard_weight, p=2, dim=-1) 
        self._easy_weight = F.normalize(easy_weight, p=2, dim=-1) 
        self._random_q = random_q
        self._loss_hn = LR.build(loss_hn)
        self._loss_en = LR.build(loss_en)

    def forward_train(
        self,
        gt_region_embeddings: torch.Tensor,
        gt_box_labels: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        gt_box_labels = torch.cat(gt_box_labels)
        batch = gt_box_labels.shape[0]
        device_labels = gt_box_labels.device
        dimdim = gt_region_embeddings.shape[-1]
        assert gt_region_embeddings.shape[0] > 0, 'no gt bboxes!!'
        assert torch.sum(gt_box_labels < 0) == 0 # 0 ~ 1202

        gt_region_embeddings = torch.nn.functional.normalize(gt_region_embeddings, p=2, dim=1)
        
        len_hard = self._hard_pad[gt_box_labels]
        len_easy = self._easy_pad[gt_box_labels]
        random_hard = []
        random_easy = []
        for limit in len_hard:
            random_hard.append(torch.randperm(limit, device=device_labels)[: self._random_q])
        for limit in len_easy:
            random_easy.append(torch.randperm(limit, device=device_labels)[: self._random_q])
        random_hard = torch.stack(random_hard) # [batch, random_q]
        random_easy = torch.stack(random_easy)
        batch_hard_weight_ = self._hard_weight[gt_box_labels].to(device=device_labels) # [b, max_hard, dim]
        random_hard_ = random_hard.unsqueeze(-1).expand(-1, -1, dimdim) # [batch, random_q, dim]
        batch_hard_weight = batch_hard_weight_.gather(1, random_hard_) # [b, random_q, dim]
        batch_easy_weight_ = self._easy_weight[gt_box_labels].to(device=device_labels)
        random_easy_ = random_easy.unsqueeze(-1).expand(-1, -1, dimdim)
        batch_easy_weight = batch_easy_weight_.gather(1, random_easy_)

        batch_anchor_weight = self._anchor_weight[gt_box_labels] # [b, dim]

        s_anchor = gt_region_embeddings @ batch_anchor_weight.to(device=device_labels).T
        s_anchor = s_anchor.gather(1, torch.arange(batch, device=device_labels).view(-1, 1)) # [b, 1]
        s_hard_neg = gt_region_embeddings @ batch_hard_weight.view(batch * self._random_q, -1).to(device=device_labels).T # [b, b * random_q]
        s_hard_neg = s_hard_neg.gather(1, torch.arange(s_hard_neg.shape[1], device=device_labels).view(-1, self._random_q)) # [b, random_q]

        s_easy_neg = gt_region_embeddings @ batch_easy_weight.view(batch * self._random_q, -1).to(device=device_labels).T # [b, b * random_q]
        s_easy_neg = s_easy_neg.gather(1, torch.arange(s_easy_neg.shape[1], device=device_labels).view(-1, self._random_q)) # [b, random_q]

        loss_hard_neg = self._loss_hn(s_anchor, s_hard_neg, with_anchor=True)
        loss_easy_neg = self._loss_en(s_hard_neg, s_easy_neg)

        return dict(loss_hard_neg=loss_hard_neg, loss_easy_neg=loss_easy_neg)


class GlobalHead(todd.Module):

    def __init__(
        self,
        *args,
        topk: int,
        classifier: todd.Config,
        loss: todd.Config,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._multilabel_topk_recall = MultilabelTopKRecall(k=topk)
        self._classifier = LINEAR_LAYERS.build(classifier)
        self._loss = LR.build(loss)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        feat = einops.reduce(feats[-1], 'b c h w -> b c', reduction='mean')
        return self._classifier(feat)

    def forward_train(
        self,
        *args,
        labels: list[torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        logits = self.forward(*args, **kwargs)
        targets = logits.new_zeros(
            logits.shape[0],
            Globals.categories.num_all,
            dtype=torch.bool,
        )
        for i, label in enumerate(labels):
            targets[i, label] = True
        return dict(
            loss_global=self._loss(logits.sigmoid(), targets),
            recall_global=self._multilabel_topk_recall(logits, targets),
        )


@DETECTORS.register_module()
class ViLD(TwoStageDetector, Student[SelfDistiller]):
    rpn_head: RPNHead
    roi_head: OADPRoIHead

    def __init__(
        self,
        *args,
        distiller: todd.Config,     
        **kwargs,
    ) -> None:
        TwoStageDetector.__init__(self, *args, **kwargs)
        Student.__init__(self, distiller)

    @property
    def num_classes(self) -> int:
        return Globals.categories.num_all

    def forward_train(
        self,
        img: torch.Tensor,
        img_metas: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        Globals.training = True
        feats = self.extract_feat(img)
        losses: dict[str, Any] = dict()
        custom_tensors: dict[str, Any] = dict()

        self._forward_train(
            feats,
            img_metas,
            losses,
            custom_tensors,
            **kwargs,
        )
        distill_losses = self.distiller(custom_tensors) 
        self.distiller.reset()
        self.distiller.step()
        losses.update(distill_losses)

        return losses

    def _forward_train(
        self,
        feats: list[torch.Tensor],
        img_metas: list[dict[str, Any]],
        losses: dict[str, Any],
        custom_tensors: dict[str, Any],
        *,
        gt_bboxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        clip_objects: list[torch.Tensor],
        object_bboxes: list[torch.Tensor],
        **kwargs,
    ) -> None:
        rpn_losses, proposals = self.rpn_head.forward_train(
            feats,
            img_metas,
            gt_bboxes,
            gt_labels=None,
            gt_bboxes_ignore=None,
            proposal_cfg=self.train_cfg.rpn_proposal,
            **kwargs,
        ) 
        losses.update(rpn_losses)
        
        roi_losses = self.roi_head.forward_train(
            feats,
            img_metas,
            proposals,
            gt_bboxes,
            gt_labels,
            None,
            **kwargs,
        )
        losses.update(roi_losses)

        self.roi_head.object_forward_train(feats, object_bboxes)
        custom_tensors['clip_objects'] = torch.cat(clip_objects).float()

@DETECTORS.register_module()
class OADP(ViLD):

    def __init__(
        self,
        *args,
        global_head: todd.Config | None = None,
        hen_head: todd.Config | None = None,
        distiller: todd.Config,
        **kwargs,
    ) -> None:
        TwoStageDetector.__init__(self, *args, **kwargs)
        if global_head is not None:
            self._global_head = GlobalHead(**global_head)
        if hen_head is not None:
            self._hen_head = HenHead(**hen_head)
        Student.__init__(self, distiller)

    @property
    def with_global(self) -> bool:
        return hasattr(self, '_global_head')

    @property
    def with_ral(self) -> bool:
        return hasattr(self, '_hen_head')

    def _forward_train(
        self,
        feats: list[torch.Tensor],
        img_metas: list[dict[str, Any]],
        losses: dict[str, Any],
        custom_tensors: dict[str, Any],
        *,
        gt_bboxes: list[torch.Tensor],
        gt_labels: list[torch.Tensor],
        clip_global: torch.Tensor | None = None,
        clip_blocks: list[torch.Tensor] | None = None,
        block_bboxes: list[torch.Tensor] | None = None,
        block_labels: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> None:
        super()._forward_train(
            feats,
            img_metas,
            losses,
            custom_tensors,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            **kwargs,
        )
        if self.with_global:
            assert clip_global is not None
            global_losses = self._global_head.forward_train(
                feats,
                labels=gt_labels,
            )
            losses.update(global_losses)
            custom_tensors['clip_global'] = clip_global.float()
        if self.roi_head.with_block:
            assert clip_blocks is not None
            assert block_bboxes is not None
            assert block_labels is not None
            block_losses = self.roi_head.block_forward_train(
                feats,
                block_bboxes,
                block_labels,
            )
            losses.update(block_losses)
            custom_tensors['clip_blocks'] = torch.cat(clip_blocks).float()
        
        if self.with_ral:
            gt_region_embeddings = self.roi_head._bbox_forward_gt_bboxes(feats, gt_bboxes)
            ral_losses = self._hen_head.forward_train(gt_region_embeddings, gt_labels)
            losses.update(ral_losses)
            
    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        Globals.training = False

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(x, proposal_list, img_metas, rescale=rescale, img=img)
