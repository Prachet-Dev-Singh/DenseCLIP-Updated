import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder
from .utils import tokenize 

class DenseCLIP(nn.Module):
    def __init__(self, class_names, context_length=5, text_dim=1024, token_embed_dim=512):
        """
        Pure PyTorch implementation of DenseCLIP aligned with official CoOp configurations.
        """
        super().__init__()
        # context_length = exactly the number of CoOp learnable prefix vectors (5)
        self.context_length = context_length  
        self.tau = 0.07
        
        # 1. Initialize Core PyTorch Modules
        self.backbone = CLIPResNetWithAttention(layers=[3, 4, 6, 3], output_dim=text_dim)
        
        # FIX: The text encoder uses the full standard CLIP window of 77 tokens
        self.text_encoder = CLIPTextContextEncoder(context_length=77, embed_dim=text_dim)
        
        self.context_decoder = ContextDecoder(
            transformer_width=256,
            transformer_heads=4,
            transformer_layers=3, 
            visual_dim=text_dim,
            dropout=0.1
        )
        
        # 2. Tokenize class names using the full 77-token window
        self.texts = torch.cat([tokenize(c, context_length=77 - self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        # 3. Setup Learnable Prompts (5 vectors of dimension 512)
        self.contexts = nn.Parameter(torch.randn(1, self.context_length, token_embed_dim))
        nn.init.trunc_normal_(self.contexts)
        
        # Gamma acts as the residual scaling factor for the vision-to-language prompting
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

    def forward(self, img):
        # 1. Extract visual features
        x = self.backbone(img)
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        
        # 2. Prepare visual context for prompting
        visual_context = torch.cat([
            global_feat.reshape(B, C, 1), 
            visual_embeddings.reshape(B, C, H*W)
        ], dim=2).permute(0, 2, 1) # Shape: [B, H*W+1, C]

        # 3. Extract text embeddings and apply Vision-to-Language Prompting
        # text_encoder dynamically injects self.contexts into the 77-token sequence
        text_embeddings = self.text_encoder(self.texts.to(img.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # 4. Compute Pixel-Text Score Maps
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        
        # Return raw score map (cosine similarity) without dividing by self.tau here
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm)
        
        # Extract the standard feature pyramid levels
        pyramid_features = list(x[0:4])
        
        return pyramid_features, score_map