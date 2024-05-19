import os
import json
import torch
import argparse
from PIL import Image
from clip import clip
from tqdm import tqdm

def main(args):
    device = 'cuda'
    model, preprocess = clip.load("ViT-B/32", device=device)

    if args.dataset == 'coco':
        ds_path = f'data/{args.dataset}/annotations/instances_{args.train_val}2017.json'
    else: # lvis
        ds_path = f'data/{args.dataset}/lvis_v1_{args.train_val}.json'
    
    ds_file = json.load(open(ds_path, 'r'))
    annotations = ds_file['annotations']
    os.makedirs(f'clip_region/{args.dataset}/{args.train_val}', exist_ok=True)
    if args.dataset == 'lvis':
        img_id_to_train_val = {x['id'] : 'train' if 'train' in x['coco_url'] else 'val' for x in ds_file['images']}

    preprocesses = []
    region_feat_paths = []
    for anno in tqdm(annotations):
        anno_id = anno['id']
        img_id = anno['image_id']
        feat_path = f'clip_region/{args.dataset}/{args.train_val}/{str(anno_id).zfill(12)}.pkl'
        if os.path.isfile(feat_path):
            continue
        if anno['bbox'][2] < 1 or anno['bbox'][3] < 1:
            continue
        region_feat_paths.append(feat_path)
        if args.dataset == 'coco':
            img = Image.open(f'data/{args.dataset}/{args.train_val}2017/{str(img_id).zfill(12)}.jpg')
        else: # lvis
            img = Image.open(f'data/{args.dataset}/{img_id_to_train_val[img_id]}2017/{str(img_id).zfill(12)}.jpg')
        bbox = (anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3])
        
        crop_img = img.crop(bbox)
        prep_img = preprocess(crop_img).unsqueeze(0).to(device)
        preprocesses.append(prep_img)

        if len(region_feat_paths) == 200:
            with torch.no_grad():
                img_feats = model.encode_image(torch.cat(preprocesses))
            for i, img_f in enumerate(img_feats):
                torch.save(img_f.cpu(), region_feat_paths[i])
            preprocesses = []
            region_feat_paths = []

    if len(region_feat_paths) > 0:
        with torch.no_grad():
            img_feats = model.encode_image(torch.cat(preprocesses))
        for i, img_f in enumerate(img_feats):
            torch.save(img_f.cpu(), region_feat_paths[i])
        preprocesses = []
        region_feat_paths = []

def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'lvis'])
    parser.add_argument('--train_val', type=str, default='val', choices=['train', 'val'])
    return parser.parse_args()

if __name__ == "__main__":
    args = parsing_argument()
    main(args)
