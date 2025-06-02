"""Helper utilities for attention visualization."""

import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Union, List
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    GPT2LMHeadModel, BertModel, RobertaModel, T5Model
)
from pathlib import Path


def load_model_and_tokenizer(
    model_name: str,
    output_attentions: bool = True,
    device: str = "auto",
    **kwargs
) -> Tuple[torch.nn.Module, Any]:
    """
    Load a transformer model and tokenizer with attention output enabled.
    
    Args:
        model_name: Name or path of the model to load
        output_attentions: Whether to enable attention output
        device: Device to load the model on
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_name, **kwargs)
    if output_attentions:
        config.output_attentions = True
    
    # Load model based on model type
    try:
        if "gpt" in model_name.lower():
            model = GPT2LMHeadModel.from_pretrained(model_name, config=config, **kwargs)
        elif "bert" in model_name.lower():
            model = BertModel.from_pretrained(model_name, config=config, **kwargs)
        elif "roberta" in model_name.lower():
            model = RobertaModel.from_pretrained(model_name, config=config, **kwargs)
        elif "t5" in model_name.lower():
            model = T5Model.from_pretrained(model_name, config=config, **kwargs)
        else:
            # Try generic AutoModel
            model = AutoModel.from_pretrained(model_name, config=config, **kwargs)
    except Exception as e:
        print(f"Warning: Could not load with specific model class, trying AutoModel: {e}")
        model = AutoModel.from_pretrained(model_name, config=config, **kwargs)
    
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def save_attention_data(
    attention_data: Dict[str, Any],
    save_path: str,
    format: str = "json",
    compress: bool = False
) -> str:
    """
    Save attention data to file.
    
    Args:
        attention_data: Dictionary containing attention data
        save_path: Path to save the data
        format: Format to save in ('json', 'pickle', 'numpy')
        compress: Whether to compress the data
        
    Returns:
        Path to the saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in attention_data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                json_data[key] = [arr.tolist() for arr in value]
            else:
                json_data[key] = value
        
        with open(save_path, 'w') as f:
            json.dump(json_data, f, indent=2)
    
    elif format == "pickle":
        with open(save_path, 'wb') as f:
            pickle.dump(attention_data, f)
    
    elif format == "numpy":
        # Save as compressed numpy archive
        np.savez_compressed(save_path, **attention_data) if compress else np.savez(save_path, **attention_data)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return save_path


def load_attention_data(file_path: str, format: Optional[str] = None) -> Dict[str, Any]:
    """
    Load attention data from file.
    
    Args:
        file_path: Path to the data file
        format: Format of the file ('json', 'pickle', 'numpy'). If None, inferred from extension.
        
    Returns:
        Dictionary containing attention data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Infer format from file extension if not provided
    if format is None:
        extension = Path(file_path).suffix.lower()
        if extension == ".json":
            format = "json"
        elif extension == ".pkl" or extension == ".pickle":
            format = "pickle"
        elif extension == ".npz":
            format = "numpy"
        else:
            raise ValueError(f"Cannot infer format from extension: {extension}")
    
    if format == "json":
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays where appropriate
        if "attention_weights" in data:
            data["attention_weights"] = [np.array(arr) for arr in data["attention_weights"]]
        
        return data
    
    elif format == "pickle":
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    elif format == "numpy":
        loaded = np.load(file_path, allow_pickle=True)
        return dict(loaded)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def validate_attention_data(attention_data: Dict[str, Any]) -> bool:
    """
    Validate attention data structure.
    
    Args:
        attention_data: Dictionary containing attention data
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ["attention_weights", "tokens", "num_layers", "num_heads", "sequence_length"]
    
    for key in required_keys:
        if key not in attention_data:
            print(f"Missing required key: {key}")
            return False
    
    # Validate attention weights structure
    attention_weights = attention_data["attention_weights"]
    if not isinstance(attention_weights, list):
        print("attention_weights should be a list")
        return False
    
    if len(attention_weights) != attention_data["num_layers"]:
        print(f"Number of attention weight layers ({len(attention_weights)}) doesn't match num_layers ({attention_data['num_layers']})")
        return False
    
    # Check first layer structure
    if len(attention_weights) > 0:
        first_layer = attention_weights[0]
        if not isinstance(first_layer, np.ndarray) or len(first_layer.shape) != 3:
            print("Each attention weight layer should be a 3D numpy array (num_heads, seq_len, seq_len)")
            return False
        
        if first_layer.shape[0] != attention_data["num_heads"]:
            print(f"Number of heads in attention weights ({first_layer.shape[0]}) doesn't match num_heads ({attention_data['num_heads']})")
            return False
    
    return True


def get_model_info(model) -> Dict[str, Any]:
    """
    Extract information about a transformer model.
    
    Args:
        model: Transformer model instance
        
    Returns:
        Dictionary with model information
    """
    config = model.config
    
    info = {
        "model_type": getattr(config, 'model_type', 'unknown'),
        "num_layers": getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 'unknown')),
        "num_heads": getattr(config, 'num_attention_heads', getattr(config, 'n_head', 'unknown')),
        "hidden_size": getattr(config, 'hidden_size', getattr(config, 'n_embd', 'unknown')),
        "vocab_size": getattr(config, 'vocab_size', 'unknown'),
        "max_position_embeddings": getattr(config, 'max_position_embeddings', getattr(config, 'n_positions', 'unknown')),
        "output_attentions": getattr(config, 'output_attentions', False)
    }
    
    return info


def create_sample_config() -> Dict[str, Any]:
    """
    Create a sample configuration dictionary.
    
    Returns:
        Sample configuration dictionary
    """
    return {
        "visualization": {
            "default_colormap": "viridis",
            "figure_size": [12, 8],
            "dpi": 300,
            "interactive": True,
            "save_format": "png"
        },
        "export": {
            "default_format": "json",
            "include_metadata": True,
            "compression": False
        },
        "model": {
            "max_length": 512,
            "batch_size": 1,
            "device": "auto",
            "cache_attention": True
        }
    }


def batch_process_texts(
    texts: List[str],
    extractor,
    batch_size: int = 8,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Process multiple texts in batches.
    
    Args:
        texts: List of texts to process
        extractor: AttentionExtractor instance
        batch_size: Number of texts to process at once
        show_progress: Whether to show progress bar
        
    Returns:
        List of attention data dictionaries
    """
    results = []
    
    if show_progress:
        try:
            from tqdm import tqdm
            text_iter = tqdm(range(0, len(texts), batch_size), desc="Processing texts")
        except ImportError:
            text_iter = range(0, len(texts), batch_size)
            print("Install tqdm for progress bars: pip install tqdm")
    else:
        text_iter = range(0, len(texts), batch_size)
    
    for i in text_iter:
        batch_texts = texts[i:i + batch_size]
        
        for text in batch_texts:
            try:
                attention_data = extractor.extract_attention_weights(text)
                results.append(attention_data)
            except Exception as e:
                print(f"Error processing text '{text[:50]}...': {e}")
                results.append(None)
    
    return results


def calculate_attention_distance_matrix(attention_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate distance matrix from attention matrix.
    
    Args:
        attention_matrix: 2D attention matrix (seq_len, seq_len)
        
    Returns:
        Distance matrix showing positional attention patterns
    """
    seq_len = attention_matrix.shape[0]
    distance_matrix = np.zeros((seq_len, seq_len))
    
    for i in range(seq_len):
        for j in range(seq_len):
            distance_matrix[i, j] = abs(i - j)
    
    return distance_matrix


def get_attention_summary_stats(attention_weights: List[np.ndarray]) -> Dict[str, float]:
    """
    Get summary statistics for attention weights.
    
    Args:
        attention_weights: List of attention matrices per layer
        
    Returns:
        Dictionary with summary statistics
    """
    all_weights = []
    for layer_weights in attention_weights:
        all_weights.extend(layer_weights.flatten())
    
    all_weights = np.array(all_weights)
    
    return {
        "mean": float(np.mean(all_weights)),
        "std": float(np.std(all_weights)),
        "min": float(np.min(all_weights)),
        "max": float(np.max(all_weights)),
        "median": float(np.median(all_weights)),
        "q25": float(np.percentile(all_weights, 25)),
        "q75": float(np.percentile(all_weights, 75))
    }


def find_supported_models() -> List[str]:
    """
    Find commonly supported transformer models for attention visualization.
    
    Returns:
        List of supported model names
    """
    return [
        "gpt2",
        "gpt2-medium", 
        "gpt2-large",
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
        "distilbert-base-uncased",
        "albert-base-v2",
        "t5-small",
        "t5-base"
    ] 