import os
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_5_6_r1epoch1_prompt.pth checkpoints/voc/fg_bg_5_5_6_voc voc')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_6_7_r1epoch1_prompt.pth checkpoints/voc/fg_bg_5_6_7_voc voc')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_7_8_r1epoch1_prompt.pth checkpoints/voc/fg_bg_5_7_8_voc voc')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_8_9_r1epoch1_prompt.pth checkpoints/voc/fg_bg_5_8_9_voc voc')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_9_10_r1epoch1_prompt.pth checkpoints/voc/fg_bg_5_9_10_voc voc')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/voc fg_bg_5_10_voc soft 0.0 0.5 0.5 checkpoints/voc/fg_bg_5_5_6_voc.pth checkpoints/voc/fg_bg_5_6_7_voc.pth checkpoints/voc/fg_bg_5_7_8_voc.pth checkpoints/voc/fg_bg_5_8_9_voc.pth checkpoints/voc/fg_bg_5_9_10_voc.pth')

os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_5_6_r1epoch1_prompt.pth checkpoints/coco/fg_bg_5_5_6_coco coco')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_6_7_r1epoch1_prompt.pth checkpoints/coco/fg_bg_5_6_7_coco coco')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_7_8_r1epoch1_prompt.pth checkpoints/coco/fg_bg_5_7_8_coco coco')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_8_9_r1epoch1_prompt.pth checkpoints/coco/fg_bg_5_8_9_coco coco')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python gen_cls_embedding.py checkpoints/exp3/fg_bg_5_9_10_r1epoch1_prompt.pth checkpoints/coco/fg_bg_5_9_10_coco coco')
os.system('export MKL_SERVICE_FORCE_INTEL=1 &&export MKL_THREADING_LAYER=GNU&& python run.py test data/lvis_clip_image_proposal_embedding/train data/lvis_clip_image_proposal_embedding/val checkpoints/coco fg_bg_5_10_coco soft 0.0 0.5 0.5 checkpoints/coco/fg_bg_5_5_6_coco.pth checkpoints/coco/fg_bg_5_6_7_coco.pth checkpoints/coco/fg_bg_5_7_8_coco.pth checkpoints/coco/fg_bg_5_8_9_coco.pth checkpoints/coco/fg_bg_5_9_10_coco.pth')
