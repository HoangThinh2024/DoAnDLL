"""
Multimodal Pill Recognition Transformer

This module implements the core multimodal transformer architecture for pharmaceutical
identification combining Vision Transformer (ViT) and BERT through cross-modal attention.

Author: DoAnDLL Team
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Any, List
import math

try:
    from transformers import BertModel, ViTModel, BertConfig, ViTConfig, AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available")

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm library not available")


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing visual and textual features.
    
    This module implements multi-head cross-attention where visual features attend
    to textual features and vice versa.
    """
    
    def __init__(self, 
                 hidden_dim: int = 768,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_dim // num_attention_heads
        
        assert hidden_dim % num_attention_heads == 0, "hidden_dim must be divisible by num_attention_heads"
        
        # Query, Key, Value projections for cross-attention
        self.query_projection = nn.Linear(hidden_dim, hidden_dim)
        self.key_projection = nn.Linear(hidden_dim, hidden_dim)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, 
                visual_features: torch.Tensor, 
                text_features: torch.Tensor,
                visual_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention
        
        Args:
            visual_features: [batch_size, visual_seq_len, hidden_dim]
            text_features: [batch_size, text_seq_len, hidden_dim]
            visual_mask: Optional attention mask for visual features
            text_mask: Optional attention mask for text features
            
        Returns:
            Tuple of attended visual and text features
        """
        batch_size = visual_features.size(0)
        
        # Reshape for multi-head attention
        def reshape_for_attention(x):
            return x.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Visual-to-text attention (visual queries attend to text keys/values)
        v_queries = reshape_for_attention(self.query_projection(visual_features))
        t_keys = reshape_for_attention(self.key_projection(text_features))
        t_values = reshape_for_attention(self.value_projection(text_features))
        
        # Compute attention scores
        v2t_scores = torch.matmul(v_queries, t_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply text mask if provided
        if text_mask is not None:
            text_mask = text_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, text_seq_len]
            v2t_scores = v2t_scores.masked_fill(text_mask == 0, float('-inf'))
        
        v2t_attention = F.softmax(v2t_scores, dim=-1)
        v2t_attention = self.dropout(v2t_attention)
        
        # Apply attention to values
        v2t_output = torch.matmul(v2t_attention, t_values)
        v2t_output = v2t_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Text-to-visual attention (text queries attend to visual keys/values)
        t_queries = reshape_for_attention(self.query_projection(text_features))
        v_keys = reshape_for_attention(self.key_projection(visual_features))
        v_values = reshape_for_attention(self.value_projection(visual_features))
        
        t2v_scores = torch.matmul(t_queries, v_keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply visual mask if provided
        if visual_mask is not None:
            visual_mask = visual_mask.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, visual_seq_len]
            t2v_scores = t2v_scores.masked_fill(visual_mask == 0, float('-inf'))
        
        t2v_attention = F.softmax(t2v_scores, dim=-1)
        t2v_attention = self.dropout(t2v_attention)
        
        t2v_output = torch.matmul(t2v_attention, v_values)
        t2v_output = t2v_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # Add residual connections and layer normalization
        attended_visual = self.layer_norm(visual_features + self.output_projection(v2t_output))
        attended_text = self.layer_norm(text_features + self.output_projection(t2v_output))
        
        return attended_visual, attended_text


class VisualEncoder(nn.Module):
    """
    Vision Transformer encoder for pill images
    """
    
    def __init__(self, 
                 model_name: str = "vit_base_patch16_224",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 output_dim: int = 768):
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        
        # Load Vision Transformer from timm
        if "vit" in model_name.lower():
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''  # Remove global pooling to get patch features
            )
            self.feature_dim = self.backbone.num_features
        else:
            # Alternative: use ResNet or other models
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool='avg'
            )
            self.feature_dim = self.backbone.num_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer to match hidden dimension
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for visual encoder
        
        Args:
            images: [batch_size, 3, 224, 224]
            
        Returns:
            Visual features: [batch_size, num_patches + 1, output_dim] for ViT
                            or [batch_size, 1, output_dim] for CNN
        """
        features = self.backbone(images)
        
        # Handle different output shapes
        if len(features.shape) == 2:  # [batch_size, feature_dim]
            features = features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features


class TextEncoder(nn.Module):
    """
    BERT encoder for pill imprint text
    """
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 output_dim: int = 768,
                 max_length: int = 128):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.output_dim = output_dim
        
        # Load BERT model
        if pretrained:
            self.backbone = BertModel.from_pretrained(model_name)
        else:
            config = BertConfig.from_pretrained(model_name)
            self.backbone = BertModel(config)
        
        self.feature_dim = self.backbone.config.hidden_size
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection layer to match hidden dimension
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for text encoder
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Text features: [batch_size, seq_len, output_dim]
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence output (all token representations)
        features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Project to desired dimension
        features = self.projection(features)
        
        return features


class FeatureFusion(nn.Module):
    """
    Feature fusion module combining visual and text features
    """
    
    def __init__(self,
                 fusion_type: str = "cross_attention",
                 hidden_dim: int = 768,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        if fusion_type == "cross_attention":
            self.cross_attention = CrossModalAttention(
                hidden_dim=hidden_dim,
                num_attention_heads=num_attention_heads,
                dropout=dropout
            )
            self.fusion_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_type == "concat":
            self.fusion_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_type == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self,
                visual_features: torch.Tensor,
                text_features: torch.Tensor,
                visual_mask: Optional[torch.Tensor] = None,
                text_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Fuse visual and text features
        
        Args:
            visual_features: [batch_size, visual_seq_len, hidden_dim]
            text_features: [batch_size, text_seq_len, hidden_dim]
            visual_mask: Optional attention mask for visual features
            text_mask: Optional attention mask for text features
            
        Returns:
            Fused features: [batch_size, hidden_dim]
        """
        if self.fusion_type == "cross_attention":
            # Apply cross-modal attention
            attended_visual, attended_text = self.cross_attention(
                visual_features, text_features, visual_mask, text_mask
            )
            
            # Global average pooling with mask consideration
            if visual_mask is not None:
                visual_mask = visual_mask.unsqueeze(-1).float()
                attended_visual = (attended_visual * visual_mask).sum(1) / visual_mask.sum(1)
            else:
                attended_visual = attended_visual.mean(1)
            
            if text_mask is not None:
                text_mask = text_mask.unsqueeze(-1).float()
                attended_text = (attended_text * text_mask).sum(1) / text_mask.sum(1)
            else:
                attended_text = attended_text.mean(1)
            
            # Concatenate and project
            fused = torch.cat([attended_visual, attended_text], dim=-1)
            fused = self.fusion_projection(fused)
            
        elif self.fusion_type == "concat":
            # Simple concatenation with pooling
            visual_pooled = visual_features.mean(1)
            text_pooled = text_features.mean(1)
            fused = torch.cat([visual_pooled, text_pooled], dim=-1)
            fused = self.fusion_projection(fused)
            
        elif self.fusion_type == "bilinear":
            # Bilinear fusion
            visual_pooled = visual_features.mean(1)
            text_pooled = text_features.mean(1)
            fused = self.bilinear(visual_pooled, text_pooled)
        
        # Apply layer norm and dropout
        fused = self.layer_norm(fused)
        fused = self.dropout(fused)
        
        return fused


class ClassificationHead(nn.Module):
    """
    Classification head for pill recognition
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dims: List[int] = [512, 256],
                 num_classes: int = 1000,
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification
        
        Args:
            features: [batch_size, input_dim]
            
        Returns:
            Logits: [batch_size, num_classes]
        """
        return self.classifier(features)


class MultimodalPillTransformer(nn.Module):
    """
    Complete multimodal transformer for pill recognition
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Initialize encoders
        visual_config = config["visual_encoder"]
        self.visual_encoder = VisualEncoder(
            model_name=visual_config["model_name"],
            pretrained=visual_config["pretrained"],
            freeze_backbone=visual_config.get("freeze_backbone", False),
            output_dim=visual_config["output_dim"]
        )
        
        text_config = config["text_encoder"]
        self.text_encoder = TextEncoder(
            model_name=text_config["model_name"],
            pretrained=text_config["pretrained"],
            freeze_backbone=text_config.get("freeze_backbone", False),
            output_dim=text_config["output_dim"],
            max_length=text_config["max_length"]
        )
        
        # Initialize fusion module
        fusion_config = config["fusion"]
        self.feature_fusion = FeatureFusion(
            fusion_type=fusion_config["type"],
            hidden_dim=fusion_config.get("hidden_dim", 768),
            num_attention_heads=fusion_config.get("num_attention_heads", 8),
            dropout=fusion_config.get("dropout", 0.1)
        )
        
        # Initialize classification head
        classifier_config = config["classifier"]
        self.classification_head = ClassificationHead(
            input_dim=fusion_config.get("hidden_dim", 768),
            hidden_dims=classifier_config["hidden_dims"],
            num_classes=classifier_config["num_classes"],
            dropout=classifier_config["dropout"]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier/Kaiming initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self,
                images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal prediction
        
        Args:
            images: [batch_size, 3, 224, 224]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing logits and optionally features
        """
        # Encode visual and text features
        visual_features = self.visual_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        
        # Create visual mask (all patches are valid for ViT)
        visual_mask = torch.ones(
            visual_features.size(0), visual_features.size(1),
            device=visual_features.device, dtype=torch.long
        )
        
        # Fuse features
        fused_features = self.feature_fusion(
            visual_features, text_features, visual_mask, attention_mask
        )
        
        # Classification
        logits = self.classification_head(fused_features)
        
        outputs = {"logits": logits}
        
        if return_features:
            outputs.update({
                "visual_features": visual_features,
                "text_features": text_features,
                "fused_features": fused_features
            })
        
        return outputs
    
    def predict(self,
                images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_probs: bool = True) -> Dict[str, torch.Tensor]:
        """
        Make predictions with probabilities
        
        Args:
            images: [batch_size, 3, 224, 224]
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            return_probs: Whether to return probabilities
            
        Returns:
            Dictionary containing predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, input_ids, attention_mask)
            logits = outputs["logits"]
            
            if return_probs:
                probs = F.softmax(logits, dim=-1)
                predictions = torch.argmax(probs, dim=-1)
                confidence = torch.max(probs, dim=-1)[0]
                
                return {
                    "predictions": predictions,
                    "probabilities": probs,
                    "confidence": confidence,
                    "logits": logits
                }
            else:
                predictions = torch.argmax(logits, dim=-1)
                return {
                    "predictions": predictions,
                    "logits": logits
                }
    
    def get_attention_weights(self,
                             images: torch.Tensor,
                             input_ids: torch.Tensor,
                             attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention weights for visualization
        """
        # This would require modifying the forward pass to return attention weights
        # Implementation depends on specific visualization needs
        pass
