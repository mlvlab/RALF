_base_ = [
    '../../datasets/ov_coco.py',
    '../../models/oadp_faster_rcnn_r50_fpn.py',
    '../../schedules/40k.py',
    '../../base.py',
]

model = dict(
    hen_head=dict(
        hen_path='ralf/neg_feature_coco.pkl',   
        random_q=10,
        loss_hn=dict(
            type='HenTripletLoss',
            margin=0.0,
            lambda_stren=1.0,
            beta=1.0,
        ),
        loss_en=dict(
            type='HenTripletLoss',
            margin=1.0,
            lambda_stren=5.0,
            beta=1.0,
        ),
    ),
    global_head=dict(
        classifier=dict(
            type='Classifier',
            prompts='data/prompts/ml_coco.pth',
            out_features=65,
        ),
    ),
    roi_head=dict(
        bbox_head=dict(
            cls_predictor_cfg=dict(
                type='ViLDClassifier',
                prompts='data/prompts/vild.pth',
            ),
        ),
        object_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
        block_head=dict(
            cls_predictor_cfg=dict(
                type='Classifier',
                prompts='data/prompts/ml_coco.pth',
            ),
        ),
    ),
)
trainer = dict(
    dataloader=dict(
        samples_per_gpu=4, 
        workers_per_gpu=4,
    ),
    optimizer=dict(
        paramwise_cfg=dict(
            custom_keys={
                'roi_head.bbox_head': dict(lr_mult=0.5),
            },
        ),
    ),
    # log_config={
    #     'interval': 50,
    #     'hooks': [{
    #         'type': 'TextLoggerHook',
    #         'by_epoch': False 
    #     }]
    # },
)
