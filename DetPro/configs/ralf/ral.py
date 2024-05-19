_base_ = '../lvis/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1_pretrain_ens.py'
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.000025)
lr_config = dict(step=[16])
total_epochs = 20
evaluation = dict(interval=20,metric=['bbox', 'segm'])
checkpoint_config = dict(interval=1, create_symlink=False)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2
    )
model = dict(
    roi_head=dict(
        use_ral=True,
        ral_path='ralf/neg_feature_lvis.pkl',
        random_q=10, 
        triplet_lambda_sim=1.0, 
        triplet_lambda_dis=10.0, 
        triplet_margin_sim=0.0,
        triplet_margin_dis=1.0, 
        triplet_beta_sim=1.0, 
        triplet_beta_dis=1.0,
    )
)