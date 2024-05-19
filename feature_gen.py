import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

from utils import extract_feature_similarity

_VAR_TH ={
    'COCO_TH': 55,
    'LVIS_TH': 800,
    'TOP_M': 2000,
}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = json.load(open(args.ds_path, 'rb'))
    v3det_sampled = json.load(open(args.sampled_path, 'r'))
    similarity, text_feature_ds, txt_feat_out = extract_feature_similarity(ds, v3det_sampled, device)

    topk_sim = []
    topk_dis = []
    for one_cat in tqdm(similarity):
        _, sim_idx = one_cat.topk(_VAR_TH['TOP_M'])
        _, dis_idx = one_cat.topk(_VAR_TH['TOP_M'], largest=False)
        topk_sim.append(sim_idx.tolist())
        topk_dis.append(dis_idx.tolist())

    counter_sim = Counter(sum(topk_sim, []))
    counter_dis = Counter(sum(topk_dis, []))

    if 'coco' in args.sampled_path:
        th = _VAR_TH['COCO_TH']
    else:
        th = _VAR_TH['LVIS_TH']

    remove_sim = [x[0] for x in counter_sim.most_common() if x[1] >= th]
    remove_dis = [x[0] for x in counter_dis.most_common() if x[1] >= th]

    output = {'feat_in' : text_feature_ds.cpu().numpy(), 'feat_out_sim' : [], 'feat_out_dis' : [], 'pad_out_sim' : [], 'pad_out_dis' : []}
    
    complement_sim = []
    for one_lvis_cat in tqdm(topk_sim):
        intersection = list(set(one_lvis_cat) & set(remove_sim))
        complement = list(set(one_lvis_cat) - set(intersection))
        complement_sim.append(complement)
    complement_dis = []
    for one_lvis_cat in tqdm(topk_dis):
        intersection = list(set(one_lvis_cat) & set(remove_dis))
        complement = list(set(one_lvis_cat) - set(intersection))
        complement_dis.append(complement)
    
    length_sim = [len(x) for x in complement_sim]
    max_sim = max(length_sim)
    length_dis = [len(x) for x in complement_dis]
    max_dis = max(length_dis)
    print(f'max_sim: {max_sim}, max_dis: {max_dis}')

    final_sim = []
    for (one_complement, one_length) in zip(complement_sim, length_sim):
        front = txt_feat_out[torch.tensor(one_complement, device=txt_feat_out.device)]
        back = torch.zeros(max_sim - one_length, 512, dtype=txt_feat_out.dtype, device=txt_feat_out.device)
        final_sim.append(torch.cat([front, back]))
    final_dis = []
    for (one_complement, one_length) in zip(complement_dis, length_dis):
        front = txt_feat_out[torch.tensor(one_complement, device=txt_feat_out.device)]
        back = torch.zeros(max_dis - one_length, 512, dtype=txt_feat_out.dtype, device=txt_feat_out.device)
        final_dis.append(torch.cat([front, back]))     

    output['feat_out_sim'] = torch.stack(final_sim).cpu().numpy()
    output['feat_out_dis'] = torch.stack(final_dis).cpu().numpy()
    output['pad_out_sim'] = np.array(length_sim)
    output['pad_out_dis'] = np.array(length_dis)

    torch.save(output, args.result_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sampled_path', default='')   # sampled_coco or sampled_lvis json path
    parser.add_argument('--ds_path', default='')        # COCO or LVIS annotation json path
    parser.add_argument('--result_path', default='')    # pickle output path
    args = parser.parse_args()

    main(args)