import torch
from torch import nn
import torch.nn.functional as F

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
    def __init__(self):
        super().__init__()

        d_model = 512
        topk_concept = 50
        self.type_embedding = nn.Parameter(torch.randn((2, d_model)))
        self.pos_embedding = nn.Parameter(torch.randn((topk_concept, d_model)))
        self.class_embedding = nn.Parameter(torch.randn((1, d_model))) # NOTE: query embedding
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=8)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=6)
        self.region_learn_send = nn.Linear(d_model, d_model) # NOTE: projection
        self.topk_concept = topk_concept

    def forward(self, x, concept_feats):
        batch, dim = x.shape
        concept_score = x @ concept_feats.T
        s_r, topk_indices = concept_score.topk(self.topk_concept) # NOTE: s_r; corresponding scores (topk_values)
        H_r = concept_feats[topk_indices.flatten()]  # NOTE: H_r; concept embeddings
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
    