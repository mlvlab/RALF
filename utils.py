import math
from tqdm import tqdm

import clip
import torch

def extract_feature_similarity(ds, v3det, device):
    clip_model, _ = clip.load('ViT-B/32', device=device)

    ds_categories = [name['name'] for name in ds['categories']]
    ds_categories = [name.replace('_', ' ') for name in ds_categories]
    prompt_ds = ['a photo of a {}'.format(name) for name in ds_categories]
    tokenized_ds = clip.tokenize(prompt_ds).to(device)

    v3det_categories = list(v3det.values())
    v3det_categories = [name.replace('_', ' ') for name in v3det_categories]
    prompt_v3det = ['a photo of a {}'.format(name) for name in v3det_categories]
    tokenized_v3det = clip.tokenize(prompt_v3det).to(device)

    with torch.no_grad():
        text_feature_ds = clip_model.encode_text(tokenized_ds)
        text_feature_v3det_ = []
        for i in tqdm(range(math.ceil(len(tokenized_v3det) / 1000))):   # NOTE: handling memeory usage
            text_feature_v3det_.append(clip_model.encode_text(tokenized_v3det[i*1000:(i+1)*1000]))
        text_feature_v3det = torch.cat(text_feature_v3det_, dim=0)

    text_feature_ds = text_feature_ds / text_feature_ds.norm(dim=-1, keepdim=True)
    text_feature_v3det = text_feature_v3det / text_feature_v3det.norm(dim=-1, keepdim=True)
    similarity = text_feature_ds @ text_feature_v3det.T

    return similarity, text_feature_ds, text_feature_v3det