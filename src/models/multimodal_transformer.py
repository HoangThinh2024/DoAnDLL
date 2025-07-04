import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import timm
from typing import Dict, Any, Optional, Tuple


class VisualEncoder(nn.Module):
    """Visual encoder using Vision Transformer or CNN backbone"""
    
    def __init__(self, model_name: str = "vit_base_patch16_224", 
                 output_dim: int = 768, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the feature dimension from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            backbone_dim = self.backbone(dummy_input).shape[-1]
        
        self.projection = nn.Linear(backbone_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch_size, 3, height, width)
        Returns:
            features: (batch_size, output_dim)
        """
        features = self.backbone(images)  # (batch_size, backbone_dim)
        features = self.projection(features)  # (batch_size, output_dim)
        features = self.layer_norm(features)
        return features


class TextEncoder(nn.Module):
    """Text encoder using BERT or similar transformer"""
    
    def __init__(self, model_name: str = "bert-base-uncased", 
                 output_dim: int = 768, max_length: int = 128):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
        # Get the hidden dimension from the model
        text_dim = self.model.config.hidden_size
        
        self.projection = nn.Linear(text_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            text_inputs: Dictionary with 'input_ids', 'attention_mask'
        Returns:
            features: (batch_size, output_dim)
        """
        outputs = self.model(**text_inputs)
        # Use CLS token representation
        features = outputs.last_hidden_state[:, 0]  # (batch_size, text_dim)
        features = self.projection(features)  # (batch_size, output_dim)
        features = self.layer_norm(features)
        return features
    
    def tokenize(self, texts: list) -> Dict[str, torch.Tensor]:
        """Tokenize text inputs"""
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism for vision-text fusion"""
    
    def __init__(self, hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: (batch_size, 1, hidden_dim)
            key: (batch_size, 1, hidden_dim)  
            value: (batch_size, 1, hidden_dim)
        Returns:
            output: (batch_size, 1, hidden_dim)
        """
        # Self-attention with residual connection
        attn_output, _ = self.multihead_attn(query, key, value)
        query = self.norm1(query + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(query)
        output = self.norm2(query + ffn_output)
        
        return output


class MultimodalFusion(nn.Module):
    """Fusion module for combining visual and textual features"""
    
    def __init__(self, fusion_type: str = "cross_attention", 
                 hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type == "cross_attention":
            self.visual_to_text_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
            self.text_to_visual_attn = CrossModalAttention(hidden_dim, num_heads, dropout)
            
        elif fusion_type == "concat":
            self.fusion_layer = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            
    def forward(self, visual_features: torch.Tensor, 
                text_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: (batch_size, hidden_dim)
            text_features: (batch_size, hidden_dim)
        Returns:
            fused_features: (batch_size, hidden_dim)
        """
        if self.fusion_type == "cross_attention":
            # Add sequence dimension for attention
            visual_seq = visual_features.unsqueeze(1)  # (batch_size, 1, hidden_dim)
            text_seq = text_features.unsqueeze(1)      # (batch_size, 1, hidden_dim)
            
            # Cross-modal attention
            v2t = self.visual_to_text_attn(visual_seq, text_seq, text_seq)
            t2v = self.text_to_visual_attn(text_seq, visual_seq, visual_seq)
            
            # Combine attended features
            fused = (v2t + t2v).squeeze(1)  # (batch_size, hidden_dim)
            
        elif self.fusion_type == "concat":
            concat_features = torch.cat([visual_features, text_features], dim=1)
            fused = self.fusion_layer(concat_features)
            
        elif self.fusion_type == "bilinear":
            fused = self.bilinear(visual_features, text_features)
            
        return fused


class MultimodalPillTransformer(nn.Module):
    """Complete multimodal transformer for pill recognition"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Visual encoder
        self.visual_encoder = VisualEncoder(
            model_name=config["visual_encoder"]["model_name"],
            output_dim=config["visual_encoder"]["output_dim"],
            pretrained=config["visual_encoder"]["pretrained"]
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=config["text_encoder"]["model_name"],
            output_dim=config["text_encoder"]["output_dim"],
            max_length=config["text_encoder"]["max_length"]
        )
        
        # Fusion module
        self.fusion = MultimodalFusion(
            fusion_type=config["fusion"]["type"],
            hidden_dim=config["fusion"]["hidden_dim"],
            num_heads=config["fusion"]["num_attention_heads"],
            dropout=config["fusion"]["dropout"]
        )
        
        # Project to fusion hidden dimension
        self.visual_proj = nn.Linear(
            config["visual_encoder"]["output_dim"],
            config["fusion"]["hidden_dim"]
        )
        self.text_proj = nn.Linear(
            config["text_encoder"]["output_dim"],
            config["fusion"]["hidden_dim"]
        )
        
        # Classifier
        classifier_layers = []
        input_dim = config["fusion"]["hidden_dim"]
        
        for hidden_dim in config["classifier"]["hidden_dims"]:
            classifier_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(config["classifier"]["dropout"])
            ])
            input_dim = hidden_dim
            
        classifier_layers.append(
            nn.Linear(input_dim, config["classifier"]["num_classes"])
        )
        
        self.classifier = nn.Sequential(*classifier_layers)
        
    def forward(self, images: torch.Tensor, text_inputs: Dict[str, torch.Tensor],
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: (batch_size, 3, height, width)
            text_inputs: Dictionary with tokenized text
            return_features: Whether to return intermediate features
        Returns:
            Dictionary with logits and optionally features
        """
        # Encode modalities
        visual_features = self.visual_encoder(images)
        text_features = self.text_encoder(text_inputs)
        
        # Project to fusion dimension
        visual_proj = self.visual_proj(visual_features)
        text_proj = self.text_proj(text_features)
        
        # Fuse modalities
        fused_features = self.fusion(visual_proj, text_proj)
        
        # Classify
        logits = self.classifier(fused_features)
        
        outputs = {"logits": logits}
        
        if return_features:
            outputs.update({
                "visual_features": visual_features,
                "text_features": text_features,
                "fused_features": fused_features
            })
            
        return outputs
    
    def get_text_tokenizer(self):
        """Get the text tokenizer for preprocessing"""
        return self.text_encoder.tokenizer
