"""Attention weight extraction utilities."""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from transformers import PreTrainedModel, PreTrainedTokenizer


class AttentionExtractor:
    """Extract and process attention weights from transformer models."""
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, device: str = "auto"):
        """
        Initialize the attention extractor.
        
        Args:
            model: Pre-trained transformer model
            tokenizer: Corresponding tokenizer
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = self._get_device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # Ensure model outputs attentions
        if hasattr(self.model.config, 'output_attentions'):
            self.model.config.output_attentions = True
    
    def _get_device(self, device: str) -> torch.device:
        """Determine the appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def extract_attention_weights(
        self, 
        text: str, 
        return_tokens: bool = True,
        max_length: Optional[int] = None
    ) -> Dict:
        """
        Extract attention weights from the model for given text.
        
        Args:
            text: Input text to analyze
            return_tokens: Whether to return tokenized text
            max_length: Maximum sequence length
            
        Returns:
            Dictionary containing attention weights, tokens, and metadata
        """
        # Tokenize input
        tokenizer_kwargs = {"return_tensors": "pt", "return_offsets_mapping": True}
        if max_length:
            tokenizer_kwargs["max_length"] = max_length
            tokenizer_kwargs["truncation"] = True
            
        tokenized = self.tokenizer(text, **tokenizer_kwargs)
        input_ids = tokenized["input_ids"].to(self.device)
        
        # Extract tokens for visualization
        if "offset_mapping" in tokenized:
            tokens = [text[s:e] for s, e in tokenized["offset_mapping"][0]]
        else:
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Get model outputs with attention
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, output_attentions=True)
        
        # Extract attention weights
        attention_weights = outputs.attentions  # Tuple of attention tensors
        
        # Convert to numpy for easier manipulation
        attention_np = []
        for layer_attention in attention_weights:
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            layer_np = layer_attention.squeeze(0).cpu().numpy()
            attention_np.append(layer_np)
        
        result = {
            "attention_weights": attention_np,
            "tokens": tokens if return_tokens else None,
            "input_ids": input_ids.cpu().numpy(),
            "num_layers": len(attention_np),
            "num_heads": attention_np[0].shape[0] if attention_np else 0,
            "sequence_length": len(tokens),
            "model_name": getattr(self.model.config, 'name_or_path', 'unknown'),
        }
        
        return result
    
    def get_attention_statistics(self, attention_weights: List[np.ndarray]) -> Dict:
        """
        Compute statistical measures for attention weights.
        
        Args:
            attention_weights: List of attention matrices per layer
            
        Returns:
            Dictionary with attention statistics
        """
        stats = {
            "layer_stats": [],
            "overall_stats": {}
        }
        
        all_weights = []
        
        for layer_idx, layer_attention in enumerate(attention_weights):
            # layer_attention shape: (num_heads, seq_len, seq_len)
            layer_mean = np.mean(layer_attention)
            layer_std = np.std(layer_attention)
            layer_entropy = self._compute_entropy(layer_attention)
            layer_sparsity = self._compute_sparsity(layer_attention)
            
            layer_stat = {
                "layer": layer_idx,
                "mean_attention": float(layer_mean),
                "std_attention": float(layer_std),
                "entropy": float(layer_entropy),
                "sparsity": float(layer_sparsity),
                "max_attention": float(np.max(layer_attention)),
                "min_attention": float(np.min(layer_attention))
            }
            stats["layer_stats"].append(layer_stat)
            all_weights.extend(layer_attention.flatten())
        
        # Overall statistics
        all_weights = np.array(all_weights)
        stats["overall_stats"] = {
            "mean_attention": float(np.mean(all_weights)),
            "std_attention": float(np.std(all_weights)),
            "entropy": float(self._compute_entropy(all_weights)),
            "sparsity": float(self._compute_sparsity(all_weights)),
            "max_attention": float(np.max(all_weights)),
            "min_attention": float(np.min(all_weights))
        }
        
        return stats
    
    def _compute_entropy(self, attention_weights: np.ndarray) -> float:
        """Compute entropy of attention distribution."""
        # Flatten and normalize
        flat_weights = attention_weights.flatten()
        # Add small epsilon to avoid log(0)
        flat_weights = flat_weights + 1e-12
        flat_weights = flat_weights / np.sum(flat_weights)
        
        # Compute entropy
        entropy = -np.sum(flat_weights * np.log(flat_weights))
        return entropy
    
    def _compute_sparsity(self, attention_weights: np.ndarray, threshold: float = 0.01) -> float:
        """Compute sparsity of attention weights."""
        total_elements = attention_weights.size
        sparse_elements = np.sum(attention_weights < threshold)
        return sparse_elements / total_elements
    
    def get_head_attention_patterns(
        self, 
        text: str, 
        layer: int, 
        head: Optional[int] = None
    ) -> Dict:
        """
        Get attention patterns for specific layer and head.
        
        Args:
            text: Input text
            layer: Layer index
            head: Head index (if None, returns all heads)
            
        Returns:
            Dictionary with attention patterns and metadata
        """
        attention_data = self.extract_attention_weights(text)
        
        if layer >= attention_data["num_layers"]:
            raise ValueError(f"Layer {layer} not available. Model has {attention_data['num_layers']} layers.")
        
        layer_attention = attention_data["attention_weights"][layer]
        
        if head is not None:
            if head >= layer_attention.shape[0]:
                raise ValueError(f"Head {head} not available. Layer has {layer_attention.shape[0]} heads.")
            head_attention = layer_attention[head]
        else:
            head_attention = layer_attention
        
        return {
            "attention_matrix": head_attention,
            "tokens": attention_data["tokens"],
            "layer": layer,
            "head": head,
            "shape": head_attention.shape
        } 