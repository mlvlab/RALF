import json
import argparse
import torch

from utils import extract_feature_similarity

_VAR_TH ={
    'COCO_TH': 13200000,
    'LVIS_TH': 12700000,
}

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = json.load(open(args.ds_path, 'rb'))
    v3det = json.load(open(args.v3det_path, 'r'))
    similarity, _, _ = extract_feature_similarity(ds, v3det, device)

    if 'coco' in args.v3det_path:
        th = _VAR_TH['COCO_TH']
    else:
        th = _VAR_TH['LVIS_TH']
    
    ds_rank = similarity.sort(descending=False)[1]
    ds_rank_var = ds_rank.to(torch.float).var(dim=0)
    ds_remove_idx = (ds_rank_var < th).nonzero(as_tuple=True)[0]

    ds_sampled = []
    for i, name in enumerate(list(v3det.values())):
        if i in ds_remove_idx:
            pass
        else:
            ds_sampled.append(name)
    ds_sampled_dict = {i: c for i, c in enumerate(ds_sampled)}

    json.dump(ds_sampled_dict, open(args.result_path, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_path', default='')        # COCO or LVIS annotation json path
    parser.add_argument('--v3det_path', default='')     # v3det_coco or v3det_lvis json path
    parser.add_argument('--result_path', default='')    # json output path
    args = parser.parse_args()

    main(args)