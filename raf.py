import os
import json
import logging
import argparse
from clip import clip
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

logger = logging.getLogger(name='RAFLog')
logger.setLevel(logging.INFO)
formatter = logging.Formatter('|%(asctime)s||%(name)s||%(levelname)s|\n%(message)s', datefmt='%Y-%m-%d %H:%M:%S')

COCO_BASES=[
    'person', 'bicycle', 'car', 'motorcycle', 'train', 'truck', 'boat',
    'bench', 'bird', 'horse', 'sheep', 'bear', 'zebra', 'giraffe',
    'backpack', 'handbag', 'suitcase', 'frisbee', 'skis', 'kite',
    'surfboard', 'bottle', 'fork', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'pizza', 'donut', 'chair',
    'bed', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'microwave',
    'oven', 'toaster', 'refrigerator', 'book', 'clock', 'vase',
    'toothbrush'
]
COCO_NOVELS=[
    'airplane', 'bus', 'cat', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
    'snowboard', 'skateboard', 'cup', 'knife', 'cake', 'couch', 'keyboard',
    'sink', 'scissors'
]
COCO_BASES_NOVELS = COCO_BASES + COCO_NOVELS

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class CBM(nn.Module): # NOTE: Augmenter
    def __init__(self, args):
        super().__init__()

        d_model = 512
        self.type_embedding = nn.Parameter(torch.randn((2, d_model)))
        self.pos_embedding = nn.Parameter(torch.randn((args.topk_concept, d_model)))
        self.class_embedding = nn.Parameter(torch.randn((1, d_model))) # NOTE: query embedding
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.region_learn_send = nn.Linear(d_model, d_model) # NOTE: projection

        concept_pkl = torch.load(args.concept_pkl_path, map_location='cuda')
        concept_feats = concept_pkl['all_noun_chunks']['text_features']
        concept_feats /= concept_feats.norm(dim=-1, keepdim=True)
        self.concept_feats = concept_feats
        self.topk_concept = args.topk_concept

    def forward(self, x):
        batch, dim = x.shape
        concept_score = x @ self.concept_feats.T
        s_r, topk_indices = concept_score.topk(self.topk_concept) # NOTE: s_r; corresponding scores (topk_values)
        H_r = self.concept_feats[topk_indices.flatten()]  # NOTE: H_r; concept embeddings
        s_r = nn.functional.softmax(s_r, dim=-1)
        corel_concept_feats = s_r.flatten().unsqueeze(1) * H_r
        kv_first = x.unsqueeze(1) + self.type_embedding[0]
        corel_concept_feats = corel_concept_feats.reshape(batch, self.topk_concept, dim)
        kv_second = corel_concept_feats + self.pos_embedding + self.type_embedding[1]
        kv = torch.cat([kv_first, kv_second], dim=1)
        
        cls_embed = self.class_embedding.expand(batch, -1, dim)
        cls_embed = cls_embed.permute(1, 0, 2)
        kv = kv.permute(1, 0, 2)

        fine = self.transformer(cls_embed, kv)
        fine = fine.permute(1, 0, 2).squeeze(1)
        fine = fine / fine.norm(dim=-1, keepdim=True)

        coarse = self.region_learn_send(x.to(torch.float32))
        coarse = coarse / coarse.norm(dim=-1, keepdim=True)

        augmented_feats = coarse + fine
        augmented_feats = augmented_feats / augmented_feats.norm(dim=-1, keepdim=True)

        return augmented_feats
    
def train_loop(dataloader, model, optimizer, loss_cls_fn=None, loss_reg_fn=None, stren_cls=1.0, stren_reg=1.0):
    size = len(dataloader.dataset)
    model.train()
    for batch, (region_feats, pseudo_cat_ids) in enumerate(tqdm(dataloader)):
        aug_feats = model(region_feats)
        losses = {}
        if loss_cls_fn is not None:
            v3det_base_txt_feats = dataloader.dataset.all_cat_txt_feats
            sim_aug = aug_feats @ v3det_base_txt_feats.to(dtype=torch.float32, device=aug_feats.device).T
            losses['loss cls'] = loss_cls_fn(sim_aug, pseudo_cat_ids) * stren_cls

        if loss_reg_fn is not None:
            losses['loss reg'] = loss_reg_fn(aug_feats, region_feats.to(torch.float32)) * stren_reg

        loss = sum(v for v in losses.values())
        losses_reduced = {k : v.item() for k, v in losses.items()}

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = loss.item(), (batch + 1) * len(region_feats)
        logger.info(f"{losses_reduced}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    sims_v_gt = []
    sims_aug_v_gt = []

    with torch.no_grad():
        for region_feats, gt_cat_txt_feats, gt_cat_ids in tqdm(dataloader):
            aug_feats = model(region_feats)

            val_txt_feats = dataloader.dataset.all_cat_txt_feats
            sim_aug_val = aug_feats @ val_txt_feats.to(torch.float32).T
            pred_cat_ids = sim_aug_val.max(dim=1)[1]

            correct += (gt_cat_ids == pred_cat_ids).type(torch.float32).sum().item()

            similarity_v_gt = region_feats @ gt_cat_txt_feats.T
            similarity_aug_v_gt = aug_feats.to(torch.float16) @ gt_cat_txt_feats.T
            sims_v_gt.append(similarity_v_gt.diag())
            sims_aug_v_gt.append(similarity_aug_v_gt.diag())

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.5f}%, Avg loss: {test_loss:>8f} \n")
    print(f'Comparison of Similarity: V_f {torch.cat(sims_v_gt).mean().item():>0.5f} +- {torch.cat(sims_v_gt).var().item():>0.9f} / Aug_v_f {torch.cat(sims_aug_v_gt).mean().item():>0.5f} +- {torch.cat(sims_aug_v_gt).var().item():>0.9f}')
    logger.info(f"Test Error: \n Accuracy: {(100 * correct):>0.5f}%, Avg loss: {test_loss:>8f} \n")
    logger.info(f'Comparison of Similarity: V_f {torch.cat(sims_v_gt).mean().item():>0.5f} +- {torch.cat(sims_v_gt).var().item():>0.9f} / Aug_v_f {torch.cat(sims_aug_v_gt).mean().item():>0.5f} +- {torch.cat(sims_aug_v_gt).var().item():>0.9f}')


class CustomAnnoDataset(Dataset):
    def __init__(self, dataset, train_val='val'):
        if dataset == 'coco':
            json_path = f'data/{dataset}/annotations/instances_{train_val}2017.json'
        else: # lvis
            json_path = f'data/{dataset}/lvis_v1_val.json'

        json_file = json.load(open(json_path, 'r'))
        annotations = json_file['annotations']        

        subfiles = os.listdir(f'clip_region/{dataset}/{train_val}')
        anno_ids = [int(filename[:-4]) for filename in subfiles]
        
        self.anno_id_to_cat_id = {x['id'] : x['category_id'] for x in annotations}

        # val
        if dataset == 'coco':
            cat_ids = [x['id'] for x in json_file['categories'] if x['name'] in COCO_BASES_NOVELS]
            cat_names = [x['name'] for x in json_file['categories'] if x['name'] in COCO_BASES_NOVELS]
            cat_id_converter = {x : i for i, x in enumerate(cat_ids)}
            anno_ids = [x for x in anno_ids if self.anno_id_to_cat_id[x] in cat_ids]
        else:
            cat_names = [x['name'] for x in json_file['categories']]
            cat_id_converter = {x['id'] : i for i, x in enumerate(json_file['categories'])}

        self.dataset = dataset
        self.train_val = train_val
        self.anno_ids = anno_ids
        self.cat_id_converter = cat_id_converter

        clip_model, _ = clip.load("ViT-B/32", device='cuda')
        text = clip.tokenize([f'a photo of {x}' for x in cat_names]).to('cuda')
        with torch.no_grad():
            txt_feats = clip_model.encode_text(text)
        txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
        self.all_cat_txt_feats = txt_feats

    def __len__(self):
        return len(self.anno_ids)

    def __getitem__(self, idx):
        anno_id = self.anno_ids[idx]
        region_feat_path = f'clip_region/{self.dataset}/{self.train_val}/{str(anno_id).zfill(12)}.pkl'
        region_feat = torch.load(region_feat_path, map_location='cuda')
        region_feat /= region_feat.norm(dim=-1, keepdim=True)
        gt_cat_id = self.anno_id_to_cat_id[anno_id]
        gt_cat_id = self.cat_id_converter[gt_cat_id]
        gt_cat_txt_f = self.all_cat_txt_feats[gt_cat_id].to(device=region_feat.device)
        return region_feat, gt_cat_txt_f, torch.tensor(gt_cat_id, device=region_feat.device)

def my_collate_fn(samples):
    xx = [x for x, _ in samples]
    region_feat = pad_sequence(xx, batch_first=True)
    xx = [x for _, x in samples]
    pseudo_cat_id = pad_sequence(xx, batch_first=True)

    region_feat = region_feat.reshape(-1, region_feat.shape[-1])
    pseudo_cat_id = pseudo_cat_id.reshape(-1)

    n_props_per_image = [x.shape[0] for x, _ in samples]
    mask = pad_sequence([torch.ones(x, dtype=torch.bool) for x in n_props_per_image], batch_first=True)
    mask = mask.reshape(-1)

    region_feat = region_feat[mask]
    pseudo_cat_id = pseudo_cat_id[mask]

    return region_feat, pseudo_cat_id

class CustomImageDataset(Dataset):
    def __init__(self, dataset, num_proposals=1000):
        if dataset == 'coco':
            json_path = f'data/{dataset}/annotations/instances_val2017.json'
        else: # lvis
            json_path = f'data/{dataset}/lvis_v1_val.json'

        json_file = json.load(open(json_path, 'r'))

        if dataset == 'coco':
            base_cat_names = COCO_BASES
        else: # lvis            
            base_cat_names = [x['name'] for x in json_file['categories'] if x['frequency'] != 'r']

        self.dataset = dataset
        self.num_proposals = num_proposals

        if os.path.exists(args.oake_file_path):
            oake_info = torch.load(args.oake_file_path, map_location='cpu')
            self.img_ids = oake_info['img_ids']
            self.img_idx_to_cat_id = oake_info['img_idx_to_cat_id']
            self.img_idx_to_filter_idx = oake_info['img_idx_to_filter_idx']
            txt_feats = oake_info['v3det_coco_lvis_gt_cat_txt_feats']
            txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
            self.all_cat_txt_feats = txt_feats
        else:
            clip_model, _ = clip.load("ViT-B/32", device='cuda')
            v3det_cat_names = json.load(open(f'v3det_{dataset}_strict.json', 'r'))
            cat_names_plus_v3det = base_cat_names + v3det_cat_names
            text = clip.tokenize([f'a photo of {x}' for x in cat_names_plus_v3det]).to('cuda')
            with torch.no_grad():
                txt_feats = clip_model.encode_text(text)
            txt_feats /= txt_feats.norm(dim=-1, keepdim=True)
            self.all_cat_txt_feats = txt_feats

            subfiles = os.listdir(f'clip_region/{dataset}_oake_object_train')
            img_ids = [int(filename[:-4]) for filename in subfiles]

            img_idx_to_cat_id = []
            img_idx_to_filter_idx = []
            for idx, subfile in enumerate(tqdm(subfiles)):
                f = torch.load(f'clip_region/{dataset}_oake_object_train/{subfile}', map_location=txt_feats.device)
                # filter low objectness score
                filter01 = f['objectness'].squeeze(1) > 0.4
                # filter too small box
                filter02 = f['bboxes'][:, 2] - f['bboxes'][:, 0] > 10
                filter03 = f['bboxes'][:, 3] - f['bboxes'][:, 1] > 10
                filter_idx = (filter01 & filter02 & filter03).nonzero().squeeze(1)
                img_idx_to_filter_idx.append(filter_idx)
                
                v_embeds = f['embeddings'][filter_idx]
                v_embeds /= v_embeds.norm(dim=-1, keepdim=True)
                sim_v_embeds = v_embeds @ txt_feats.T
                img_idx_to_cat_id.append(sim_v_embeds.argmax(dim=-1))

            oake_info = {'img_ids' : img_ids, 'img_idx_to_cat_id' : img_idx_to_cat_id, 'img_idx_to_filter_idx' : img_idx_to_filter_idx, 'v3det_coco_lvis_gt_cat_txt_feats' : txt_feats}
            torch.save(oake_info, args.oake_file_path)
        
            self.img_ids = img_ids
            self.img_idx_to_cat_id = img_idx_to_cat_id
            self.img_idx_to_filter_idx = img_idx_to_filter_idx

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        region_proposal_path = f'clip_region/{self.dataset}_oake_object_train/{str(img_id).zfill(12)}.pth'
        pth_file = torch.load(region_proposal_path, map_location='cuda')
        filter_idx = self.img_idx_to_filter_idx[idx][:self.num_proposals]
        region_feat = pth_file['embeddings'][filter_idx]
        region_feat /= region_feat.norm(dim=-1, keepdim=True)
        pseudo_cat_id = self.img_idx_to_cat_id[idx][:self.num_proposals].to(device=region_feat.device)
        return region_feat, pseudo_cat_id

def main(args):
    if os.path.isdir(args.work_dir):
        print(f'path "{args.work_dir}" already exists.')
    else:
        os.makedirs(args.work_dir, exist_ok=True)
        print(f'Args\n-------------------------------\n{args}\n-------------------------------')
        logger.info(f'Args\n-------------------------------\n{args}\n-------------------------------')

        train_data = CustomImageDataset(dataset=args.dataset, num_proposals=args.num_proposals)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)

        test_data = CustomAnnoDataset(dataset=args.dataset, train_val='val')
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

        model = CBM(args)
        model.to('cuda')

        loss_cls_fn = nn.CrossEntropyLoss(reduction='mean')
        loss_reg_fn = nn.MSELoss(reduction='mean')
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)        

        file_handler = logging.FileHandler(f'{args.work_dir}/output.log', mode='w') 
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        for t in range(args.epochs):
            print(f"Epoch {t + 1}\n-------------------------------")
            logger.info(f"Epoch {t + 1}\n-------------------------------")
            train_loop(train_dataloader, model, optimizer, loss_cls_fn, loss_reg_fn, args.stren_cls, args.stren_reg)
            test_loop(test_dataloader, model)
        torch.save(model.state_dict(), f'{args.work_dir}/weight_{args.epochs}.pth')
        print("Done!")
        logger.info('Done!')

def parsing_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'lvis'])
    parser.add_argument('--concept_pkl_path', type=str, default='') 
    parser.add_argument('--oake_file_path', type=str, default='')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--topk_concept', type=int, default=50)
    parser.add_argument('--num_proposals', type=int, default=30)
    parser.add_argument('--stren_cls', type=float, default=5.0)
    parser.add_argument('--stren_reg', type=float, default=1.0)
    parser.add_argument('--work_dir', type=str, default='output')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parsing_argument()
    main(args)