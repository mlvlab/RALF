import os.path as osp
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu():
    url = clip._MODELS["ViT-B/32"]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))
    
    try:
        # loading JIT archive
        print("jit version")
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
    
    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):

    def __init__(self, classnames, clip_model, random_init, ctx_init = None, bg_class = False, ctx=8, cls_token_position='end'):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = ctx
        self.class_token_position = cls_token_position
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, \
            f'cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})'
        
        if random_init:
            # random init
            print('Initializing a generic context')
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx)
            print(f'Initial context: "{prompt_prefix}"')
            print(f'Number of context words (tokens): {n_ctx}')
        else:
            # use given words to initialize context vectors
            # ctx_init = "This is a photo of"
            # n_ctx = len(ctx_init.split(' '))
            # prompt = clip.tokenize(ctx_init)
            # with torch.no_grad():
            #     embedding = clip_model.token_embedding(prompt).type(dtype)
            # ctx_vectors = embedding[0, 1:1+n_ctx, :]
            # prompt_prefix = ctx_init
            print('Load context')
            ctx_vectors = ctx_init
            n_ctx = ctx_init.shape[0]
            prompt_prefix = ' '.join(['X'] * n_ctx)
            print(f'Initial context: "{prompt_prefix}"')
            print(f'Number of context words (tokens): {n_ctx}')


        

        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        classnames = [name.replace('_', ' ') for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + ' ' + name + '.' for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        
        # print('debug?? ', tokenized_prompts[:, 1+n_ctx:], embedding[:, 1+n_ctx:, :])
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer('token_prefix', embedding[:, :1, :]) # SOS
        self.register_buffer('token_suffix', embedding[:, 1+n_ctx:, :]) # CLS, EOS
        # print("DEBUG")
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts # torch.Tensor
        self.name_lens = name_lens

        self.clip_model = clip_model

        self.bg_class = bg_class
        bg_prompt = torch.empty(10, ctx_dim, dtype=dtype)
        nn.init.normal_(bg_prompt, std=0.02)
        self.bg_prompt = nn.Parameter(bg_prompt)

        tokenized_bg_prompt = ' '.join(['X'] * 10)#(self.n_ctx+2))
        self.tokenized_bg_prompt = clip.tokenize(tokenized_bg_prompt)
    
    def get_bg_prompt(self):
        # print(self.token_prefix.device)
        # print(self.ctx.device)
        # print(self.bg_prompt.device)
        # with torch.no_grad():
        #     embedding = self.clip_model.token_embedding(tokenized_prompts).type(dtype)
        prompt = torch.cat([
            self.token_prefix[0], # (1, dim)
            # self.ctx, # (n_ctx, dim)
            self.bg_prompt, 
            self.token_suffix[2][2:] # (*, dim)
        ], dim=0)
        return prompt, self.tokenized_bg_prompt
    
    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        
        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.class_token_position == 'end':
            prompts = torch.cat([
                prefix, # (n_cls, 1, dim)
                ctx, # (n_cls, n_ctx, dim)
                suffix # (n_cls, *, dim)
            ], dim=1)
        elif self.class_token_position == 'middle':
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat([
                    prefix_i, # (1, 1, dim)
                    ctx_i_half1, # (1, n_ctx//2, dim)
                    class_i, # (1, name_len, dim)
                    ctx_i_half2, # (1, n_ctx//2, dim)
                    suffix_i # (1, *, dim)
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == 'front':
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat([
                    prefix_i, # (1, 1, dim)
                    class_i, # (1, name_len, dim)
                    ctx_i, # (1, n_ctx, dim)
                    suffix_i # (1, *, dim)
                ], dim=1)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        
        else:
            raise ValueError
        

        if self.bg_class:
            bg_prompt, bg_token = self.get_bg_prompt()
            prompts = torch.cat([prompts, bg_prompt[None]])
            tokenized_prompts = torch.cat([self.tokenized_prompts, bg_token])

            return prompts, tokenized_prompts
        else:
            return prompts, self.tokenized_prompts
    
    def forward_for_classes(self, classes):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(len(classes), -1, -1)
        
        n_ctx = self.n_ctx
        prompt_prefix = ' '.join(['X'] * n_ctx)
        prompts = [prompt_prefix + ' ' + name + '.' for name in classes]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        # tokenized_prompts = tokenized_prompts.cuda()
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts.cuda()).type(self.clip_model.dtype)
        prefix = embedding[:, :1, :] # SOS
        suffix = embedding[:, 1+n_ctx:, :] # CLS, EOS

        prompts = torch.cat([
            prefix, # (n_cls, 1, dim)
            ctx, # (n_cls, n_ctx, dim)
            suffix # (n_cls, *, dim)
        ], dim=1)
        
        if self.bg_class:
            bg_prompt, bg_token = self.get_bg_prompt()
            prompts = torch.cat([prompts, bg_prompt[None]])
            tokenized_prompts = torch.cat([tokenized_prompts, bg_token])

        return prompts, tokenized_prompts


class CustomCLIP(nn.Module):

    def __init__(self, classnames, clip_model, random_init, ctx_init = None, bg_class = False,ctx=8,cls_token_position='end'):
        super().__init__()
        self.prompt_learner = PromptLearner(classnames, clip_model, random_init, ctx_init, bg_class,ctx,cls_token_position=cls_token_position)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        
        # parallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.text_encoder = nn.DataParallel(self.text_encoder)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # bg_vectors = torch.empty(1, 512, dtype=self.dtype)
        # nn.init.normal_(bg_vectors, std=0.02)
        # self.bg_embedding = nn.Parameter(bg_vectors) # background
    
    def get_embedding(self):
        prompts, tokenized_prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, tokenized_prompts)

        # text_features = torch.cat([text_features, self.bg_embedding], dim = 0)
        return text_features

    def forward(self, image_features):
        text_features = self.get_embedding()

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = image_features @ text_features.t()
        return logits

