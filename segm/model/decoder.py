import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_
from transformers import GPT2Model, GPT2Config

from segm.model.blocks import Block, FeedForward
from segm.model.utils import init_weights


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class MaskTransformer(nn.Module):
    def __init__(
        self,
        n_cls,
        patch_size,
        d_encoder,
        n_layers,
        n_heads,
        d_model,
        d_ff,
        drop_path_rate,
        dropout,
        use_gpt2_init=False,
        gpt2_model_name="gpt2",
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        # GPT-2 initialization for transformer blocks if requested
        if use_gpt2_init:
            # Load pretrained GPT-2 model
            gpt2_model = GPT2Model.from_pretrained(gpt2_model_name)
            
            # Create blocks and initialize from GPT-2
            self.blocks = nn.ModuleList([
                Block(d_model, n_heads, d_ff, dropout, 0.0) 
                for _ in range(n_layers)
            ])
            
            # Copy weights from GPT-2 blocks to our blocks
            # Note: We'll only copy weights if there are enough layers in the GPT-2 model
            num_copy_layers = min(n_layers, len(gpt2_model.h))
            print(f"Copying {num_copy_layers} layers from GPT-2")
            for i in range(num_copy_layers):
                # Get GPT-2 block
                gpt2_block = gpt2_model.h[i]
                
                # Copy attention weights (query, key, value projections)
                # GPT-2 stores q,k,v in a single matrix, we need to split them
                if self.blocks[i].attn.qkv.weight.shape[0] == gpt2_block.attn.c_attn.weight.shape[0] * 3:
                    # Split the GPT-2 qkv weight and copy
                    gpt2_qkv = gpt2_block.attn.c_attn.weight
                    gpt2_qkv_bias = gpt2_block.attn.c_attn.bias
                    
                    qkv_size = gpt2_qkv.shape[0]
                    q_size = k_size = v_size = qkv_size // 3
                    
                    # Our block expects a single qkv matrix
                    self.blocks[i].attn.qkv.weight.data.copy_(gpt2_qkv.t())
                    self.blocks[i].attn.qkv.bias.data.copy_(gpt2_qkv_bias)
                
                # Copy MLP weights if dimensions match
                if self.blocks[i].mlp.fc1.weight.shape == (gpt2_block.mlp.c_fc.weight.shape[1], gpt2_block.mlp.c_fc.weight.shape[0]):
                    self.blocks[i].mlp.fc1.weight.data.copy_(gpt2_block.mlp.c_fc.weight.t())
                    self.blocks[i].mlp.fc1.bias.data.copy_(gpt2_block.mlp.c_fc.bias)
                
                if self.blocks[i].mlp.fc2.weight.shape == (gpt2_block.mlp.c_proj.weight.shape[1], gpt2_block.mlp.c_proj.weight.shape[0]):
                    self.blocks[i].mlp.fc2.weight.data.copy_(gpt2_block.mlp.c_proj.weight.t())
                    self.blocks[i].mlp.fc2.bias.data.copy_(gpt2_block.mlp.c_proj.bias)
                
                # Copy layer norm weights
                if self.blocks[i].norm1.weight.shape == gpt2_block.ln_1.weight.shape:
                    self.blocks[i].norm1.weight.data.copy_(gpt2_block.ln_1.weight)
                    self.blocks[i].norm1.bias.data.copy_(gpt2_block.ln_1.bias)
                
                if self.blocks[i].norm2.weight.shape == gpt2_block.ln_2.weight.shape:
                    self.blocks[i].norm2.weight.data.copy_(gpt2_block.ln_2.weight)
                    self.blocks[i].norm2.bias.data.copy_(gpt2_block.ln_2.bias)
                
            print(f"Initialized {num_copy_layers} transformer blocks with GPT-2 weights")
        else:
            # Original initialization with dropout path
            print("Original initialization with dropout path")
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
            self.blocks = nn.ModuleList(
                [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
            )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        # Apply weight initialization only to components other than the blocks if using GPT-2
        if use_gpt2_init:
            # Initialize only non-transformer parts
            self.apply(lambda module: init_weights(module) 
                      if not isinstance(module, Block) else None)
        else:
            # Original initialization for everything
            self.apply(init_weights)
            
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls :]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)
