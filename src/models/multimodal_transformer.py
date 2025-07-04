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
from transformers import BertModel, ViTModel, BertConfig, ViTConfig, AutoModel, AutoTokenizer
import timm
from typing import Dict, Tuple, Optional, Any
import math


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
                query_features: torch.Tensor,
                key_value_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query_features: Features that will attend (B, seq_len_q, hidden_dim)
            key_value_features: Features to attend to (B, seq_len_kv, hidden_dim)
            attention_mask: Mask for attention weights (B, seq_len_q, seq_len_kv)
            
        Returns:
            attended_features: Features after cross-attention (B, seq_len_q, hidden_dim)
            attention_weights: Attention weights (B, num_heads, seq_len_q, seq_len_kv)
        """
        batch_size, seq_len_q, _ = query_features.shape
        seq_len_kv = key_value_features.shape[1]
        
        # Project to Q, K, V
        Q = self.query_projection(query_features)  # (B, seq_len_q, hidden_dim)
        K = self.key_projection(key_value_features)  # (B, seq_len_kv, hidden_dim)
        V = self.value_projection(key_value_features)  # (B, seq_len_kv, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len_q, self.num_attention_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_attention_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, V)  # (B, num_heads, seq_len_q, head_dim)
        
        # Reshape and project
        attended_values = attended_values.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.hidden_dim
        )
        
        # Output projection with residual connection and layer norm
        output = self.output_projection(attended_values)
        output = self.layer_norm(output + query_features)
        
        return output, attention_weights


class VisualEncoder(nn.Module):
    """
    Vision Transformer encoder for processing pill images.
    Enhanced version with better configuration and feature extraction.
    """
    
    def __init__(self, 
                 model_name: str = "google/vit-base-patch16-224",
                 output_dim: int = 768, 
                 pretrained: bool = True,
                 freeze_layers: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Try to load ViT model from transformers, fallback to timm
        try:
            if pretrained:
                self.vit = ViTModel.from_pretrained(model_name)
                self.feature_dim = self.vit.config.hidden_size
                self.use_transformers = True
            else:
                config = ViTConfig.from_pretrained(model_name)
                self.vit = ViTModel(config)
                self.feature_dim = config.hidden_size
                self.use_transformers = True
        except:
            # Fallback to timm
            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                self.feature_dim = self.backbone(dummy_input).shape[-1]
            self.use_transformers = False
        
        # Freeze specified number of layers
        if freeze_layers > 0 and self.use_transformers:
            for i, layer in enumerate(self.vit.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        # Projection layer if needed
        if self.feature_dim != output_dim:
            self.projection = nn.Linear(self.feature_dim, output_dim)
        else:
            self.projection = nn.Identity()
            
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for visual encoding.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            
        Returns:
            Dictionary containing visual features and sequence output
        """
        if self.use_transformers:
            outputs = self.vit(pixel_values=pixel_values)
            sequence_output = outputs.last_hidden_state  # (B, num_patches + 1, hidden_dim)
            pooled_output = sequence_output[:, 0]  # (B, hidden_dim) - [CLS] token
        else:
            features = self.backbone(pixel_values)  # (B, feature_dim)
            pooled_output = features
            sequence_output = features.unsqueeze(1)  # Add sequence dimension
        
        # Apply projection, dropout and layer norm
        pooled_output = self.projection(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        # Process sequence output
        if sequence_output.dim() == 3:
            sequence_output = self.projection(sequence_output)
            sequence_output = self.dropout(sequence_output)
        
        return {
            'sequence_output': sequence_output,
            'pooled_output': pooled_output
        }


class TextEncoder(nn.Module):
    """
    BERT encoder for processing text imprints on pills.
    Enhanced version with better tokenization and feature extraction.
    """
    
    def __init__(self,
                 model_name: str = "bert-base-uncased",
                 output_dim: int = 768, 
                 max_length: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        
        # Load BERT model and tokenizer
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Get the hidden dimension from the model
        text_dim = self.model.config.hidden_size
        
        # Projection layer if needed
        if text_dim != output_dim:
            self.projection = nn.Linear(text_dim, output_dim)
        else:
            self.projection = nn.Identity()
            
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for text encoding.
        
        Args:
            input_ids: Token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            token_type_ids: Token type IDs (B, seq_len)
            
        Returns:
            Dictionary containing text features and sequence output
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get sequence output and pooled output
        sequence_output = outputs.last_hidden_state  # (B, seq_len, hidden_dim)
        
        # For pooled output, use [CLS] token or mean pooling
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Mean pooling with attention mask
            pooled_output = (sequence_output * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        # Apply projection, dropout and layer norm
        pooled_output = self.projection(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        # Process sequence output
        sequence_output = self.projection(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        return {
            'sequence_output': sequence_output,
            'pooled_output': pooled_output,
            'attention_mask': attention_mask
        }


class FusionLayer(nn.Module):
    """
    Fusion layer that combines visual and textual features through cross-modal attention.
    """
    
    def __init__(self,
                 hidden_dim: int = 768,
                 num_attention_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Cross-modal attention layers
        self.visual_to_text_attention = CrossModalAttention(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        self.text_to_visual_attention = CrossModalAttention(
            hidden_dim=hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Feature fusion layers
        self.fusion_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self,
                visual_features: torch.Tensor,
                text_features: torch.Tensor,
                text_attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for feature fusion.
        
        Args:
            visual_features: Visual features (B, num_patches, hidden_dim)
            text_features: Text features (B, seq_len, hidden_dim)
            text_attention_mask: Text attention mask (B, seq_len)
            
        Returns:
            Dictionary containing fused features and attention weights
        """
        # Cross-modal attention: visual attends to text
        visual_attended, v2t_attention = self.visual_to_text_attention(
            query_features=visual_features,
            key_value_features=text_features,
            attention_mask=text_attention_mask.unsqueeze(1) if text_attention_mask is not None else None
        )
        
        # Cross-modal attention: text attends to visual
        text_attended, t2v_attention = self.text_to_visual_attention(
            query_features=text_features,
            key_value_features=visual_features
        )
        
        # Pool attended features
        if visual_attended.dim() == 3:
            visual_pooled = visual_attended[:, 0]  # Use [CLS]-like token
        else:
            visual_pooled = visual_attended
            
        if text_attended.dim() == 3:
            text_pooled = text_attended.mean(dim=1)  # Average pooling
        else:
            text_pooled = text_attended
        
        # Concatenate and fuse features
        fused_features = torch.cat([visual_pooled, text_pooled], dim=-1)
        fused_features = self.fusion_projection(fused_features)
        fused_features = self.layer_norm(fused_features)
        
        return {
            'fused_features': fused_features,
            'visual_attended': visual_attended,
            'text_attended': text_attended,
            'v2t_attention': v2t_attention,
            't2v_attention': t2v_attention
        }


class ClassificationHead(nn.Module):
    """
    Classification head for pill recognition.
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dims: list = [1536, 512],
                 num_classes: int = 1000,
                 dropout: float = 0.2):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Final classification layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for classification.
        
        Args:
            features: Input features (B, input_dim)
            
        Returns:
            logits: Classification logits (B, num_classes)
        """
        return self.classifier(features)


class MultimodalPillTransformer(nn.Module):
    """
    Complete multimodal transformer for pill recognition.
    
    This model combines visual and textual information through cross-modal attention
    to perform accurate pharmaceutical identification.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Extract configuration
        visual_config = config.get('visual_encoder', {})
        text_config = config.get('text_encoder', {})
        fusion_config = config.get('fusion', {})
        classifier_config = config.get('classifier', {})
        
        # Get dimensions
        hidden_dim = fusion_config.get('hidden_dim', 768)
        
        # Initialize encoders
        self.visual_encoder = VisualEncoder(
            model_name=visual_config.get('model_name', 'google/vit-base-patch16-224'),
            output_dim=hidden_dim,
            pretrained=visual_config.get('pretrained', True),
            freeze_layers=visual_config.get('freeze_layers', 0),
            dropout=visual_config.get('dropout', 0.1)
        )
        
        self.text_encoder = TextEncoder(
            model_name=text_config.get('model_name', 'bert-base-uncased'),
            output_dim=hidden_dim,
            max_length=text_config.get('max_length', 128),
            dropout=text_config.get('dropout', 0.1)
        )
        
        # Initialize fusion layer
        self.fusion_layer = FusionLayer(
            hidden_dim=hidden_dim,
            num_attention_heads=fusion_config.get('num_attention_heads', 8),
            dropout=fusion_config.get('dropout', 0.1)
        )
        
        # Initialize classification head
        self.classification_head = ClassificationHead(
            input_dim=hidden_dim,
            hidden_dims=classifier_config.get('hidden_dims', [1536, 512]),
            num_classes=classifier_config.get('num_classes', 1000),
            dropout=classifier_config.get('dropout', 0.2)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for custom layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multimodal pill recognition.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            input_ids: Text token IDs (B, seq_len)
            attention_mask: Text attention mask (B, seq_len)
            token_type_ids: Text token type IDs (B, seq_len)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing logits, probabilities, and optionally attention weights
        """
        # Encode visual features
        visual_outputs = self.visual_encoder(pixel_values)
        visual_features = visual_outputs['sequence_output']
        
        # Encode text features
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        text_features = text_outputs['sequence_output']
        
        # Fuse features through cross-modal attention
        fusion_outputs = self.fusion_layer(
            visual_features=visual_features,
            text_features=text_features,
            text_attention_mask=attention_mask
        )
        fused_features = fusion_outputs['fused_features']
        
        # Classification
        logits = self.classification_head(fused_features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Prepare output
        outputs = {
            'logits': logits,
            'probabilities': probabilities,
            'visual_features': visual_outputs['pooled_output'],
            'text_features': text_outputs['pooled_output'],
            'fused_features': fused_features
        }
        
        # Add attention weights if requested
        if return_attention:
            outputs.update({
                'v2t_attention': fusion_outputs['v2t_attention'],
                't2v_attention': fusion_outputs['t2v_attention']
            })
        
        return outputs
    
    def predict(self,
                pixel_values: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                token_type_ids: Optional[torch.Tensor] = None,
                top_k: int = 5) -> Dict[str, Any]:
        """
        Make predictions with confidence scores and top-k results.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            input_ids: Text token IDs (B, seq_len)
            attention_mask: Text attention mask (B, seq_len)
            token_type_ids: Text token type IDs (B, seq_len)
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary containing predictions, confidence scores, and top-k results
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_attention=True
            )
            
            probabilities = outputs['probabilities']
            
            # Get top-k predictions
            top_k_probs, top_k_indices = torch.topk(probabilities, k=top_k, dim=-1)
            
            # Get primary prediction
            predicted_class = top_k_indices[:, 0]
            confidence = top_k_probs[:, 0]
            
            return {
                'predicted_class': predicted_class.cpu().numpy(),
                'confidence': confidence.cpu().numpy(),
                'top_k_predictions': {
                    'classes': top_k_indices.cpu().numpy(),
                    'probabilities': top_k_probs.cpu().numpy()
                },
                'attention_weights': {
                    'visual_to_text': outputs['v2t_attention'].cpu().numpy(),
                    'text_to_visual': outputs['t2v_attention'].cpu().numpy()
                }
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'MultimodalPillTransformer',
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'visual_encoder': self.visual_encoder.model_name,
            'text_encoder': self.text_encoder.model_name,
            'hidden_dim': self.config.get('fusion', {}).get('hidden_dim', 768),
            'num_classes': self.config.get('classifier', {}).get('num_classes', 1000)
        }


def create_model(config: Dict[str, Any]) -> MultimodalPillTransformer:
    """
    Factory function to create a multimodal pill transformer model.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized MultimodalPillTransformer model
    """
    return MultimodalPillTransformer(config)


# Example usage and testing
if __name__ == "__main__":
    # Example configuration
    config = {
        'visual_encoder': {
            'model_name': 'google/vit-base-patch16-224',
            'pretrained': True,
            'freeze_layers': 0,
            'dropout': 0.1
        },
        'text_encoder': {
            'model_name': 'bert-base-uncased',
            'max_length': 128,
            'dropout': 0.1
        },
        'fusion': {
            'hidden_dim': 768,
            'num_attention_heads': 8,
            'dropout': 0.1
        },
        'classifier': {
            'hidden_dims': [1536, 512],
            'num_classes': 1000,
            'dropout': 0.2
        }
    }
    
    # Create model
    model = create_model(config)
    
    # Print model info
    info = model.get_model_info()
    print("Model Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy data
    batch_size = 2
    pixel_values = torch.randn(batch_size, 3, 224, 224)
    input_ids = torch.randint(0, 1000, (batch_size, 128))
    attention_mask = torch.ones(batch_size, 128)
    
    # Forward pass
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_attention=True
    )
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # Test prediction
    predictions = model.predict(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        top_k=5
    )
    
    print(f"\nPrediction results:")
    print(f"  Predicted classes: {predictions['predicted_class']}")
    print(f"  Confidence scores: {predictions['confidence']}")
        
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
