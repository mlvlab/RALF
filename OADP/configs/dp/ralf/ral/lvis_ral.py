ann_file_prefix = 'data/lvis_v1/annotations/'
categories = 'lvis'
cls_predictor_cfg = {
    'type': 'ViLDClassifier',
    'prompts': 'data/prompts/detpro_lvis.pth',
    'scaler': {
        'train': 0.01,
        'val': 0.007
    }
}
data_root = 'data/lvis_v1/'
dataset_type = 'OV_LVIS'
model = {
    'type': 'OADP',
    'backbone': {
        'type': 'ResNet',
        'depth': 50,
        'num_stages': 4,
        'out_indices': (0, 1, 2, 3),
        'frozen_stages': 1,
        'norm_cfg': {
            'type': 'BN',
            'requires_grad': True
        },
        'norm_eval': True,
        'style': 'caffe',
        'init_cfg': None
    },
    'neck': {
        'type': 'FPN',
        'in_channels': [256, 512, 1024, 2048],
        'out_channels': 256,
        'norm_cfg': {
            'type': 'SyncBN',
            'requires_grad': True
        },
        'num_outs': 5
    },
    'hen_head':{
        'hen_path': 'ralf/neg_feature_lvis.pkl',
        'random_q': 10,
        'loss_hn': {
            'type': 'HenTripletLoss',
            'margin': 0.0,
            'lambda_stren': 1.0,
            'beta': 1.0,
        },
        'loss_en': {
            'type': 'HenTripletLoss',
            'margin': 1.0,
            'lambda_stren': 10.0,
            'beta': 1.0,
        },
    },
    'rpn_head': {
        'type': 'RPNHead',
        'in_channels': 256,
        'feat_channels': 256,
        'anchor_generator': {
            'type': 'AnchorGenerator',
            'scales': [8],
            'ratios': [0.5, 1.0, 2.0],
            'strides': [4, 8, 16, 32, 64]
        },
        'bbox_coder': {
            'type': 'DeltaXYWHBBoxCoder',
            'target_means': [0.0, 0.0, 0.0, 0.0],
            'target_stds': [1.0, 1.0, 1.0, 1.0]
        },
        'loss_cls': {
            'type': 'CrossEntropyLoss',
            'use_sigmoid': True,
            'loss_weight': 1.0
        },
        'loss_bbox': {
            'type': 'L1Loss',
            'loss_weight': 1.0
        }
    },
    'roi_head': {
        'type': 'OADPRoIHead',
        'bbox_roi_extractor': {
            'type': 'SingleRoIExtractor',
            'roi_layer': {
                'type': 'RoIAlign',
                'output_size': 7,
                'sampling_ratio': 0
            },
            'out_channels': 256,
            'featmap_strides': [4, 8, 16, 32]
        },
        'bbox_head': {
            'type': 'Shared4Conv1FCBBoxHead',
            'in_channels': 256,
            'fc_out_channels': 1024,
            'roi_feat_size': 7,
            'bbox_coder': {
                'type': 'DeltaXYWHBBoxCoder',
                'target_means': [0.0, 0.0, 0.0, 0.0],
                'target_stds': [0.1, 0.1, 0.2, 0.2]
            },
            'reg_class_agnostic': True,
            'norm_cfg': {
                'type': 'SyncBN',
                'requires_grad': True
            },
            'loss_cls': {
                'type': 'CrossEntropyLoss',
                'use_sigmoid': False,
                'loss_weight': 1.0
            },
            'loss_bbox': {
                'type': 'L1Loss',
                'loss_weight': 1.0
            },
            'num_classes': None,
            'cls_predictor_cfg': {
                'type': 'ViLDClassifier',
                'prompts': 'data/prompts/detpro_lvis.pth',
                'scaler': {
                    'train': 0.01,
                    'val': 0.007
                }
            }
        },
        'object_head': {
            'type': 'Shared4Conv1FCObjectBBoxHead',
            'cls_predictor_cfg': {
                'type': 'ViLDClassifier',
                'prompts': 'data/prompts/detpro_lvis.pth',
                'scaler': {
                    'train': 0.01,
                    'val': 0.007
                }
            }
        },
        'block_head': {
            'type': 'Shared2FCBlockBBoxHead',
            'topk': 5,
            'loss': {
                'type': 'AsymmetricLoss',
                'weight': {
                    'type': 'WarmupScheduler',
                    'gain': 16,
                    'end': 1000
                },
                'gamma_neg': 4,
                'gamma_pos': 0
            },
            'cls_predictor_cfg': {
                'type': 'ViLDClassifier',
                'prompts': 'data/prompts/detpro_lvis.pth',
                'scaler': {
                    'train': 0.01,
                    'val': 0.007
                }
            }
        },
        'mask_roi_extractor': {
            'type': 'SingleRoIExtractor',
            'roi_layer': {
                'type': 'RoIAlign',
                'output_size': 14,
                'sampling_ratio': 0
            },
            'out_channels': 256,
            'featmap_strides': [4, 8, 16, 32]
        },
        'mask_head': {
            'type': 'FCNMaskHead',
            'num_convs': 4,
            'in_channels': 256,
            'conv_out_channels': 256,
            'class_agnostic': True,
            'loss_mask': {
                'type': 'CrossEntropyLoss',
                'use_mask': True,
                'loss_weight': 1.0
            },
            'num_classes': None
        }
    },
    'train_cfg': {
        'rpn': {
            'assigner': {
                'type': 'MaxIoUAssigner',
                'pos_iou_thr': 0.7,
                'neg_iou_thr': 0.3,
                'min_pos_iou': 0.3,
                'match_low_quality': True,
                'ignore_iof_thr': -1
            },
            'sampler': {
                'type': 'RandomSampler',
                'num': 256,
                'pos_fraction': 0.5,
                'neg_pos_ub': -1,
                'add_gt_as_proposals': False
            },
            'allowed_border': -1,
            'pos_weight': -1,
            'debug': False
        },
        'rpn_proposal': {
            'nms_pre': 2000,
            'max_per_img': 1000,
            'nms': {
                'type': 'nms',
                'iou_threshold': 0.7
            },
            'min_bbox_size': 0
        },
        'rcnn': {
            'assigner': {
                'type': 'MaxIoUAssigner',
                'pos_iou_thr': 0.5,
                'neg_iou_thr': 0.5,
                'min_pos_iou': 0.5,
                'match_low_quality': False,
                'ignore_iof_thr': -1
            },
            'sampler': {
                'type': 'RandomSampler',
                'num': 512,
                'pos_fraction': 0.25,
                'neg_pos_ub': -1,
                'add_gt_as_proposals': True
            },
            'pos_weight': -1,
            'debug': False,
            'mask_size': 28
        }
    },
    'test_cfg': {
        'rpn': {
            'nms_pre': 1000,
            'max_per_img': 1000,
            'nms': {
                'type': 'nms',
                'iou_threshold': 0.7
            },
            'min_bbox_size': 0
        },
        'rcnn': {
            'score_thr': 0.0,
            'nms': {
                'type': 'nms',
                'iou_threshold': 0.5
            },
            'max_per_img': 300,
            'mask_thr_binary': 0.5
        }
    },
    'distiller': {
        'type': 'SelfDistiller',
        'student_hooks': {
            'objects': {
                'inputs': (),
                'action': {
                    'type': 'StandardHook',
                    'path': '.roi_head._object_head.fc_cls._linear'
                }
            },
            'blocks': {
                'inputs': (),
                'action': {
                    'type': 'StandardHook',
                    'path': '.roi_head._block_head.fc_cls._linear'
                }
            }
        },
        'adapts': {},
        'losses': {
            'loss_clip_objects': {
                'inputs': ('objects', 'clip_objects'),
                'action': {
                    'type': 'L1Loss',
                    'weight': {
                        'type': 'WarmupScheduler',
                        'gain': 256,
                        'end': 200
                    }
                }
            },
            'loss_clip_blocks': {
                'inputs': ('blocks', 'clip_blocks'),
                'action': {
                    'type': 'L1Loss',
                    'weight': {
                        'type': 'WarmupScheduler',
                        'gain': 128,
                        'end': 200
                    }
                }
            },
            'loss_clip_block_relations': {
                'inputs': ('blocks', 'clip_blocks'),
                'action': {
                    'type': 'RKDLoss',
                    'weight': {
                        'type': 'WarmupScheduler',
                        'gain': 8,
                        'end': 200
                    }
                }
            }
        }
    }
}
norm = {
    'mean': [123.675, 116.28, 103.53],
    'std': [58.395, 57.12, 57.375],
    'to_rgb': True
}
oake_root = 'data/lvis_v1/oake/'
trainer = {
    'dataloader': {
        'samples_per_gpu': 2,
        'workers_per_gpu': 2,
        'dataset': {
            'type': 'ClassBalancedDataset',
            'oversample_thr': 0.001,
            'dataset': {
                'type':
                'OV_LVIS',
                'img_prefix':
                'data/lvis_v1/',
                'ann_file':
                'data/lvis_v1/annotations/lvis_v1_train.866.json',
                'pipeline': [{
                    'type': 'LoadImageFromFile'
                }, {
                    'type': 'LoadAnnotations',
                    'with_bbox': True,
                    'with_mask': True
                }, {
                    'type': 'LoadCLIPFeatures',
                    'default': {
                        'task_name': 'train2017',
                        'type': 'PthAccessLayer'
                    },
                    'globals_': {
                        'data_root': 'data/lvis_v1/oake/globals'
                    },
                    'blocks': {
                        'data_root': 'data/lvis_v1/oake/blocks'
                    },
                    'objects': {
                        'data_root': 'data/lvis_v1/oake/objects'
                    }
                }, {
                    'type': 'Resize',
                    'img_scale': [(1330, 640), (1333, 800)],
                    'multiscale_mode': 'range',
                    'keep_ratio': True
                }, {
                    'type': 'RandomFlip',
                    'flip_ratio': 0.5
                }, {
                    'type': 'Normalize',
                    'mean': [123.675, 116.28, 103.53],
                    'std': [58.395, 57.12, 57.375],
                    'to_rgb': True
                }, {
                    'type': 'Pad',
                    'size_divisor': 32
                }, {
                    'type': 'DefaultFormatBundle'
                }, {
                    'type':
                    'ToTensor',
                    'keys': ['block_bboxes', 'block_labels', 'object_bboxes']
                }, {
                    'type':
                    'ToDataContainer',
                    'fields': [{
                        'key': 'clip_blocks'
                    }, {
                        'key': 'block_bboxes'
                    }, {
                        'key': 'block_labels'
                    }, {
                        'key': 'clip_objects'
                    }, {
                        'key': 'object_bboxes'
                    }]
                }, {
                    'type':
                    'Collect',
                    'keys': [
                        'img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                        'clip_global', 'clip_blocks', 'block_bboxes',
                        'block_labels', 'clip_objects', 'object_bboxes'
                    ]
                }]
            }
        }
    },
    'optimizer': {
        'type': 'SGD',
        'lr': 0.02,
        'momentum': 0.9,
        'weight_decay': 2.5e-05
    },
    'optimizer_config': {
        'grad_clip': None
    },
    'lr_config': {
        'policy': 'step',
        'warmup': 'linear',
        'warmup_iters': 500,
        'warmup_ratio': 0.001,
        'step': [16, 19]
    },
    'runner': {
        'type': 'EpochBasedRunner',
        'max_epochs': 24
    },
    'workflow': [('train', 1)],
    'checkpoint_config': {
        'interval': 1
    },
    'evaluation': {
        'interval': 24
    },
    'log_config': {
        'interval': 50,
        'hooks': [{
            'type': 'TextLoggerHook',
            'by_epoch': False
        }]
    },
    'custom_hooks': [{
        'type': 'NumClassCheckHook'
    }],
    'fp16': {
        'loss_scale': {
            'init_scale': 64.0
        }
    },
    'log_level': 'INFO',
    'resume_from': None,
    'load_from': 'pretrained/soco/soco_star_mask_rcnn_r50_fpn_400e.pth',
    'seed': 3407,
    'gpu_ids': range(0, 8),
    'device': 'cuda'
}
validator = {
    'dataloader': {
        'samples_per_gpu': 1,
        'workers_per_gpu': 2,
        'dataset': {
            'type':
            'OV_LVIS',
            'ann_file':
            'data/lvis_v1/annotations/lvis_v1_val.1203.json',
            'img_prefix':
            'data/lvis_v1/',
            'pipeline': [{
                'type': 'LoadImageFromFile'
            }, {
                'type':
                'MultiScaleFlipAug',
                'img_scale': (1333, 800),
                'flip':
                False,
                'transforms': [{
                    'type': 'Resize',
                    'keep_ratio': True
                }, {
                    'type': 'RandomFlip'
                }, {
                    'type': 'Normalize',
                    'mean': [123.675, 116.28, 103.53],
                    'std': [58.395, 57.12, 57.375],
                    'to_rgb': True
                }, {
                    'type': 'Pad',
                    'size_divisor': 32
                }, {
                    'type': 'ImageToTensor',
                    'keys': ['img']
                }, {
                    'type': 'Collect',
                    'keys': ['img']
                }]
            }]
        }
    },
    'fp16': False
}
