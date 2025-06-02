"""Main attention visualization class."""

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Union, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer

try:
    import inspectus
    INSPECTUS_AVAILABLE = True
except ImportError:
    INSPECTUS_AVAILABLE = False

from .extractor import AttentionExtractor
from ..utils.config import Config


class AttentionVisualizer:
    """Main class for visualizing attention patterns in transformer models."""
    
    def __init__(
        self, 
        model: PreTrainedModel, 
        tokenizer: PreTrainedTokenizer, 
        device: str = "auto",
        config: Optional[Config] = None
    ):
        """
        Initialize the attention visualizer.
        
        Args:
            model: Pre-trained transformer model
            tokenizer: Corresponding tokenizer  
            device: Device to run inference on
            config: Configuration object for visualization settings
        """
        self.extractor = AttentionExtractor(model, tokenizer, device)
        self.config = config or Config()
        
        # Set up visualization defaults
        plt.style.use('default')
        sns.set_palette("viridis")
    
    def visualize_attention(
        self,
        text: str,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        save_path: Optional[str] = None,
        interactive: bool = True,
        **kwargs
    ) -> None:
        """
        Main visualization method that creates attention plots.
        
        Args:
            text: Input text to visualize
            layer: Specific layer to visualize (if None, shows all layers)
            head: Specific head to visualize (if None, shows all heads)
            save_path: Path to save the visualization
            interactive: Whether to create interactive plots
            **kwargs: Additional visualization parameters
        """
        if interactive:
            self._create_interactive_visualization(text, layer, head, save_path, **kwargs)
        else:
            self._create_static_visualization(text, layer, head, save_path, **kwargs)
    
    def _create_interactive_visualization(
        self,
        text: str,
        layer: Optional[int],
        head: Optional[int], 
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """Create interactive plotly-based visualization."""
        attention_data = self.extractor.extract_attention_weights(text)
        
        if layer is not None:
            # Single layer visualization
            self._plot_single_layer_interactive(attention_data, layer, head, save_path, **kwargs)
        else:
            # Multi-layer overview
            self._plot_multi_layer_interactive(attention_data, save_path, **kwargs)
    
    def _create_static_visualization(
        self,
        text: str,
        layer: Optional[int],
        head: Optional[int],
        save_path: Optional[str],
        **kwargs
    ) -> None:
        """Create static matplotlib-based visualization."""
        attention_data = self.extractor.extract_attention_weights(text)
        
        if layer is not None:
            self._plot_single_layer_static(attention_data, layer, head, save_path, **kwargs)
        else:
            self._plot_multi_layer_static(attention_data, save_path, **kwargs)
    
    def plot_attention_heatmap(
        self,
        text: str,
        layer: int,
        head: int,
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Generate attention heatmap for specific layer and head.
        
        Args:
            text: Input text
            layer: Layer index
            head: Head index
            title: Plot title
            figsize: Figure size
            save_path: Path to save the plot
            **kwargs: Additional matplotlib parameters
        """
        head_data = self.extractor.get_head_attention_patterns(text, layer, head)
        attention_matrix = head_data["attention_matrix"]
        tokens = head_data["tokens"]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(
            attention_matrix, 
            cmap=kwargs.get('cmap', 'viridis'),
            aspect='auto'
        )
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        # Set title
        if title is None:
            title = f"Attention Heatmap - Layer {layer}, Head {head}"
        ax.set_title(title, fontsize=14)
        
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_layers(
        self,
        text: str,
        layers: List[int],
        head: Optional[int] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Compare attention patterns across multiple layers.
        
        Args:
            text: Input text
            layers: List of layer indices to compare
            head: Specific head to compare (if None, averages across heads)
            save_path: Path to save the comparison plot
            **kwargs: Additional plotting parameters
        """
        attention_data = self.extractor.extract_attention_weights(text)
        tokens = attention_data["tokens"]
        
        n_layers = len(layers)
        fig, axes = plt.subplots(1, n_layers, figsize=(5 * n_layers, 8))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, layer in enumerate(layers):
            layer_attention = attention_data["attention_weights"][layer]
            
            if head is not None:
                attention_matrix = layer_attention[head]
            else:
                # Average across heads
                attention_matrix = np.mean(layer_attention, axis=0)
            
            im = axes[idx].imshow(
                attention_matrix,
                cmap=kwargs.get('cmap', 'viridis'),
                aspect='auto'
            )
            
            axes[idx].set_title(f"Layer {layer}")
            axes[idx].set_xticks(range(len(tokens)))
            axes[idx].set_yticks(range(len(tokens)))
            axes[idx].set_xticklabels(tokens, rotation=45, ha='right')
            axes[idx].set_yticklabels(tokens)
            
            plt.colorbar(im, ax=axes[idx])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_attention_stats(self, text: str) -> Dict:
        """
        Get quantitative attention metrics.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with attention statistics
        """
        attention_data = self.extractor.extract_attention_weights(text)
        return self.extractor.get_attention_statistics(attention_data["attention_weights"])
    
    def analyze_head_specialization(
        self,
        texts: List[str],
        layer: int,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Analyze what different heads focus on across multiple texts.
        
        Args:
            texts: List of texts to analyze
            layer: Layer to analyze
            save_path: Path to save analysis results
            
        Returns:
            Dictionary with head specialization analysis
        """
        head_patterns = {}
        
        for text in texts:
            attention_data = self.extractor.extract_attention_weights(text)
            layer_attention = attention_data["attention_weights"][layer]
            
            for head_idx in range(layer_attention.shape[0]):
                if head_idx not in head_patterns:
                    head_patterns[head_idx] = []
                
                head_attention = layer_attention[head_idx]
                # Compute head-specific metrics
                entropy = self.extractor._compute_entropy(head_attention)
                sparsity = self.extractor._compute_sparsity(head_attention)
                max_attention = np.max(head_attention)
                
                head_patterns[head_idx].append({
                    "text": text,
                    "entropy": entropy,
                    "sparsity": sparsity,
                    "max_attention": max_attention
                })
        
        # Aggregate statistics per head
        head_stats = {}
        for head_idx, patterns in head_patterns.items():
            entropies = [p["entropy"] for p in patterns]
            sparsities = [p["sparsity"] for p in patterns]
            max_attentions = [p["max_attention"] for p in patterns]
            
            head_stats[head_idx] = {
                "mean_entropy": np.mean(entropies),
                "std_entropy": np.std(entropies),
                "mean_sparsity": np.mean(sparsities),
                "std_sparsity": np.std(sparsities),
                "mean_max_attention": np.mean(max_attentions),
                "std_max_attention": np.std(max_attentions)
            }
        
        return {
            "layer": layer,
            "head_statistics": head_stats,
            "raw_patterns": head_patterns
        }
    
    def plot_head_specialization(self, head_analysis: Dict, save_path: Optional[str] = None) -> None:
        """
        Plot head specialization analysis results.
        
        Args:
            head_analysis: Results from analyze_head_specialization
            save_path: Path to save the plot
        """
        head_stats = head_analysis["head_statistics"]
        
        heads = list(head_stats.keys())
        entropies = [head_stats[h]["mean_entropy"] for h in heads]
        sparsities = [head_stats[h]["mean_sparsity"] for h in heads]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Entropy plot
        ax1.bar(heads, entropies)
        ax1.set_xlabel("Head Index")
        ax1.set_ylabel("Mean Entropy")
        ax1.set_title(f"Attention Entropy by Head (Layer {head_analysis['layer']})")
        
        # Sparsity plot
        ax2.bar(heads, sparsities)
        ax2.set_xlabel("Head Index")
        ax2.set_ylabel("Mean Sparsity")
        ax2.set_title(f"Attention Sparsity by Head (Layer {head_analysis['layer']})")
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_attention_data(
        self,
        text: str,
        format: str = "json",
        save_path: Optional[str] = None
    ) -> Union[Dict, str]:
        """
        Export attention weights and metadata.
        
        Args:
            text: Input text
            format: Export format ('json', 'csv', 'numpy')
            save_path: Path to save the exported data
            
        Returns:
            Exported data or path to saved file
        """
        attention_data = self.extractor.extract_attention_weights(text)
        stats = self.extractor.get_attention_statistics(attention_data["attention_weights"])
        
        export_data = {
            "text": text,
            "tokens": attention_data["tokens"],
            "model_name": attention_data["model_name"],
            "num_layers": attention_data["num_layers"],
            "num_heads": attention_data["num_heads"],
            "sequence_length": attention_data["sequence_length"],
            "statistics": stats
        }
        
        if format == "json":
            import json
            # Convert numpy arrays to lists for JSON serialization
            export_data["attention_weights"] = [arr.tolist() for arr in attention_data["attention_weights"]]
            
            if save_path:
                with open(save_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                return save_path
            return export_data
        
        elif format == "numpy":
            if save_path:
                np.savez(save_path, **{
                    "attention_weights": attention_data["attention_weights"],
                    "metadata": export_data
                })
                return save_path
            return attention_data["attention_weights"]
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def use_inspectus_visualization(
        self,
        text: str,
        chart_types: Optional[List[str]] = None,
        color_scheme: Union[str, Dict] = "viridis",
        save_dir: Optional[str] = None
    ) -> None:
        """
        Use inspectus library for advanced attention visualization.
        
        Args:
            text: Input text to visualize
            chart_types: List of chart types to generate
            color_scheme: Color scheme for plots
            save_dir: Directory to save plots
        """
        if not INSPECTUS_AVAILABLE:
            raise ImportError("inspectus library is required for this visualization. Install with: pip install inspectus")
        
        attention_data = self.extractor.extract_attention_weights(text)
        tokens = attention_data["tokens"]
        
        # Convert back to torch tensors for inspectus
        attention_tensors = []
        for layer_attention in attention_data["attention_weights"]:
            tensor = torch.from_numpy(layer_attention).unsqueeze(0)  # Add batch dimension
            attention_tensors.append(tensor)
        
        if chart_types is None:
            chart_types = [
                'attention_matrix',
                'query_token_heatmap', 
                'key_token_heatmap',
                'dimension_heatmap',
                'token_dim_heatmap'
            ]
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
            for chart_type in chart_types:
                plt.clf()
                
                inspectus.attention(
                    attention_tensors,
                    tokens,
                    chart_types=[chart_type],
                    color=color_scheme if isinstance(color_scheme, str) else color_scheme.get(chart_type, 'viridis')
                )
                
                save_path = os.path.join(save_dir, f"attention_{chart_type}.png")
                plt.savefig(save_path, bbox_inches='tight', dpi=300)
                plt.close()
            
            print(f"✅ Saved attention visualizations to: {save_dir}")
        else:
            # Display interactive plot
            inspectus.attention(
                attention_tensors,
                tokens,
                chart_types=chart_types,
                color=color_scheme
            )
            plt.show()
    
    def _plot_single_layer_interactive(self, attention_data: Dict, layer: int, head: Optional[int], save_path: Optional[str], **kwargs) -> None:
        """Create interactive single layer visualization with plotly."""
        layer_attention = attention_data["attention_weights"][layer]
        tokens = attention_data["tokens"]
        
        if head is not None:
            attention_matrix = layer_attention[head]
            title = f"Attention Heatmap - Layer {layer}, Head {head}"
        else:
            attention_matrix = np.mean(layer_attention, axis=0)
            title = f"Attention Heatmap - Layer {layer} (averaged across heads)"
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_matrix,
            x=tokens,
            y=tokens,
            colorscale='Viridis',
            showscale=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Key Tokens",
            yaxis_title="Query Tokens",
            width=800,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def _plot_multi_layer_interactive(self, attention_data: Dict, save_path: Optional[str], **kwargs) -> None:
        """Create interactive multi-layer overview with plotly."""
        # Create subplots for multiple layers
        n_layers = min(4, attention_data["num_layers"])  # Show first 4 layers
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f"Layer {i}" for i in range(n_layers)],
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        tokens = attention_data["tokens"]
        
        for i in range(n_layers):
            layer_attention = attention_data["attention_weights"][i]
            # Average across heads
            avg_attention = np.mean(layer_attention, axis=0)
            
            row = i // 2 + 1
            col = i % 2 + 1
            
            fig.add_trace(
                go.Heatmap(
                    z=avg_attention,
                    x=tokens,
                    y=tokens,
                    colorscale='Viridis',
                    showscale=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Multi-Layer Attention Overview",
            width=1000,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
    
    def _plot_single_layer_static(self, attention_data: Dict, layer: int, head: Optional[int], save_path: Optional[str], **kwargs) -> None:
        """Create static single layer visualization with matplotlib."""
        self.plot_attention_heatmap(
            attention_data["tokens"], 
            layer, 
            head or 0, 
            save_path=save_path,
            **kwargs
        )
    
    def _plot_multi_layer_static(self, attention_data: Dict, save_path: Optional[str], **kwargs) -> None:
        """Create static multi-layer visualization with matplotlib."""
        n_layers = min(4, attention_data["num_layers"])
        layers = list(range(n_layers))
        
        # Use the existing compare_layers method
        text = " ".join(attention_data["tokens"])  # Reconstruct text
        self.compare_layers(text, layers, save_path=save_path, **kwargs) 