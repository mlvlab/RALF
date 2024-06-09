import os
import cv2
import time
import mmcv
import todd
import torch
import argparse
import numpy as np
from tqdm import tqdm

from ..base import Globals
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmdet.models import build_detector
from mmdet.core import encode_mask_results
from mmdet.datasets.pipelines import Compose
from mmcv.image import tensor2imgs

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=todd.Config.load)
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--image', help='Path to the image file')
    parser.add_argument('--video', help='Path to the video file')
    parser.add_argument('--output_dir', default='demo/output', help='Path to the demo output file')
    parser.add_argument('--override', action=todd.DictAction)
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--score_thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args

def build_model(config: todd.Config, checkpoint: str, device: str) -> torch.nn.Module:
    config.model.pop('train_cfg', None)
    config.model.pop('pretrained', None)
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
    model.cfg = config  

    model.CLASSES = Globals.categories.all_
    model = model.to(device)
    model.eval()
    return model

def prepare_data(img, cfg, device):
    """Prepare data for inference."""
    if isinstance(cfg.data, dict):
        cfg.data = mmcv.ConfigDict(cfg.data)
    
    if 'test' not in cfg.data:
        cfg.data['test'] = mmcv.ConfigDict(
            dict(
                type='OV_LVIS',
                pipeline=[
                    dict(type='LoadImageFromWebcam'),
                    dict(
                        type='MultiScaleFlipAug',
                        img_scale=(1333, 800),
                        flip=False,
                        transforms=[
                            dict(type='Resize', keep_ratio=True),
                            dict(type='RandomFlip'),
                            dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
                            dict(type='Pad', size_divisor=32),
                            dict(type='ImageToTensor', keys=['img']),
                            dict(type='Collect', keys=['img']),
                        ]
                    )
                ]
            )
        )

    if 'img_norm_cfg' not in cfg:
        cfg.img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

    test_pipeline = Compose(cfg.data.test.pipeline)
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if device != 'cpu':
        data = scatter(data, [device])[0]
    return data

def process_image(model, img, score_thr=0.3, output_path=None):
    data = prepare_data(img, model.cfg, next(model.parameters()).device)
    model.eval()
    results = []

    with torch.no_grad():
        img = [data['img'][0].unsqueeze(0)]  
        result = model(return_loss=False, rescale=True, img=img, img_metas=data['img_metas'])
        result = [(bbox_results, encode_mask_results(mask_results)) for bbox_results, mask_results in result]
        results.extend(result)

    img_metas = data['img_metas'][0][0]
    img_tensor = data['img'][0]
    imgs = tensor2imgs(img_tensor.unsqueeze(0), **img_metas['img_norm_cfg'])  
    for img, img_meta in zip(imgs, [img_metas]):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        bboxes = np.vstack([res for res in results[0][0] if res.size > 0])
        labels = np.hstack([np.full(res.shape[0], i, dtype=np.int32) for i, res in enumerate(results[0][0]) if res.size > 0])

        mmcv.imshow_det_bboxes(
            img_show,
            bboxes,
            labels,
            bbox_color = 'green',
            text_color = 'blue',
            thickness = 2,
            font_scale = 0.5,
            class_names=model.CLASSES,
            score_thr=score_thr,
            show=False,
            out_file=output_path
        )

    print("Inference complete.")

def process_video(model, video_path, score_thr=0.3, output_path=None):
    video_reader = mmcv.VideoReader(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = None

    for frame in tqdm(video_reader):
        data = prepare_data(frame, model.cfg, next(model.parameters()).device)
        results = []
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, img=[data['img'][0].unsqueeze(0)] , img_metas=data['img_metas'])
            results.extend(result)

        img_tensor = data['img'][0]
        imgs = tensor2imgs(img_tensor.unsqueeze(0), **model.cfg.img_norm_cfg)  
        img_show = imgs[0]

        bboxes = np.vstack([res for res in results[0][0] if res.size > 0])
        labels = np.hstack([np.full(res.shape[0], i, dtype=np.int32) for i, res in enumerate(results[0][0]) if res.size > 0])

        img_show = mmcv.imresize(img_show, (frame.shape[1], frame.shape[0]))

        if video_writer is None:
            height, width = frame.shape[:2]
            video_writer = cv2.VideoWriter(output_path, fourcc, video_reader.fps, (width, height))

        img_show = mmcv.imshow_det_bboxes(
            img_show,
            bboxes,
            labels,
            bbox_color = 'green',
            text_color = 'blue',
            thickness = 2,
            font_scale = 0.5,
            class_names=model.CLASSES,
            score_thr=score_thr,
            show=False
        )
        video_writer.write(img_show)

    video_writer.release()
    print("Inference complete.")

def main() -> None:
    args = parse_args()
    config: todd.Config = args.config
    if args.override is not None:
        config.override(args.override)

    from ..base import coco, lvis 
    Globals.categories = eval(config.categories)
    past_time = time.time()

    model = build_model(config, args.checkpoint, device=args.device)
    if args.image:
        img = mmcv.imread(args.image)
        output_path = os.path.join(args.output_dir, args.image.split('/')[-1])
        process_image(model, img, score_thr=args.score_thr, output_path=output_path)
    elif args.video:
        output_path = os.path.join(args.output_dir, args.video.split('/')[-1])
        process_video(model, args.video, score_thr=args.score_thr, output_path=output_path)
    else:
        raise ValueError("Either --image or --video must be specified")
    print(f'Time: {time.time() - past_time}')

if __name__ == '__main__':
    main()