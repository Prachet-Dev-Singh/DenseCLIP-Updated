import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import CLIPResNetWithAttention, CLIPTextContextEncoder, ContextDecoder
from .utils import tokenize 

class DenseCLIP(nn.Module):
    def __init__(self, class_names, context_length=5, text_dim=1024):
        """
        Pure PyTorch implementation of DenseCLIP.
        """
        super().__init__()
        self.context_length = context_length
        self.tau = 0.07
        
        # 1. Initialize Core PyTorch Modules
        # Note: Defaults to ResNet-50 configurations. 
        # For ResNet-101 or ViT, adjust the 'layers' and 'output_dim' accordingly.
        self.backbone = CLIPResNetWithAttention(layers=[3, 4, 6, 3], output_dim=text_dim)
        self.text_encoder = CLIPTextContextEncoder(context_length=13, embed_dim=text_dim)
        self.context_decoder = ContextDecoder(visual_dim=text_dim)
        
        # 2. Tokenize class names
        self.texts = torch.cat([tokenize(c, context_length=self.context_length) for c in class_names])
        self.num_classes = len(self.texts)

        # 3. Setup Learnable Prompts (Context-Aware Prompting)
        learnable_context_length = self.text_encoder.context_length - self.context_length
        self.contexts = nn.Parameter(torch.randn(1, learnable_context_length, 512))
        nn.init.trunc_normal_(self.contexts)
        
        # Gamma acts as the residual scaling factor for the vision-to-language prompting
        self.gamma = nn.Parameter(torch.ones(text_dim) * 1e-4)

    def forward(self, img):
        # 1. Extract visual features
        # x contains the multi-scale feature pyramid + the final attention pool output
        x = self.backbone(img)
        global_feat, visual_embeddings = x[4]
        B, C, H, W = visual_embeddings.shape
        
        # 2. Prepare visual context for prompting
        visual_context = torch.cat([
            global_feat.reshape(B, C, 1), 
            visual_embeddings.reshape(B, C, H*W)
        ], dim=2).permute(0, 2, 1) # Shape: [B, H*W+1, C]

        # 3. Extract text embeddings and apply Vision-to-Language Prompting
        text_embeddings = self.text_encoder(self.texts.to(img.device), self.contexts).expand(B, -1, -1)
        text_diff = self.context_decoder(text_embeddings, visual_context)
        text_embeddings = text_embeddings + self.gamma * text_diff

        # 4. Compute Pixel-Text Score Maps
        visual_embeddings_norm = F.normalize(visual_embeddings, dim=1, p=2)
        text_norm = F.normalize(text_embeddings, dim=2, p=2)
        
        # Einsum performs the dense matching: visual[B, C, H, W] * text[B, K, C] -> score[B, K, H, W]
        score_map = torch.einsum('bchw,bkc->bkhw', visual_embeddings_norm, text_norm) / self.tau
        
        # Extract the standard feature pyramid levels (typically used in FPN)
        pyramid_features = list(x[0:4])
        
        return pyramid_features, score_map