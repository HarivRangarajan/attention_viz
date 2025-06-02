"""Advanced analysis tools for attention patterns."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

from .extractor import AttentionExtractor


class AttentionAnalyzer:
    """Advanced analysis tools for attention patterns."""
    
    def __init__(self, extractor: AttentionExtractor):
        """
        Initialize the attention analyzer.
        
        Args:
            extractor: AttentionExtractor instance for getting attention data
        """
        self.extractor = extractor
    
    def analyze_attention_flow(self, text: str, layer: int) -> Dict:
        """
        Analyze attention flow patterns in a specific layer.
        
        Args:
            text: Input text to analyze
            layer: Layer index to analyze
            
        Returns:
            Dictionary with attention flow analysis
        """
        attention_data = self.extractor.extract_attention_weights(text)
        
        if layer >= attention_data["num_layers"]:
            raise ValueError(f"Layer {layer} not available. Model has {attention_data['num_layers']} layers.")
        
        layer_attention = attention_data["attention_weights"][layer]
        tokens = attention_data["tokens"]
        num_heads, seq_len, _ = layer_attention.shape
        
        # Calculate attention flow metrics
        flow_analysis = {
            "layer": layer,
            "tokens": tokens,
            "num_heads": num_heads,
            "sequence_length": seq_len,
            "head_flows": [],
            "aggregate_flows": {}
        }
        
        # Analyze each head's attention flow
        for head_idx in range(num_heads):
            head_attention = layer_attention[head_idx]
            
            # Calculate outgoing attention (how much each token attends to others)
            outgoing_attention = np.sum(head_attention, axis=1)
            
            # Calculate incoming attention (how much attention each token receives)
            incoming_attention = np.sum(head_attention, axis=0)
            
            # Find attention hubs (tokens that receive high attention)
            attention_hubs = np.argsort(incoming_attention)[-3:]  # Top 3
            
            # Find attention sources (tokens that give high attention)
            attention_sources = np.argsort(outgoing_attention)[-3:]  # Top 3
            
            head_flow = {
                "head": head_idx,
                "outgoing_attention": outgoing_attention.tolist(),
                "incoming_attention": incoming_attention.tolist(),
                "attention_hubs": [int(idx) for idx in attention_hubs],
                "attention_sources": [int(idx) for idx in attention_sources],
                "max_attention_value": float(np.max(head_attention)),
                "attention_concentration": float(np.std(head_attention))
            }
            
            flow_analysis["head_flows"].append(head_flow)
        
        # Aggregate analysis across all heads
        all_outgoing = np.mean([flow["outgoing_attention"] for flow in flow_analysis["head_flows"]], axis=0)
        all_incoming = np.mean([flow["incoming_attention"] for flow in flow_analysis["head_flows"]], axis=0)
        
        flow_analysis["aggregate_flows"] = {
            "average_outgoing": all_outgoing.tolist(),
            "average_incoming": all_incoming.tolist(),
            "global_hubs": [int(idx) for idx in np.argsort(all_incoming)[-3:]],
            "global_sources": [int(idx) for idx in np.argsort(all_outgoing)[-3:]]
        }
        
        return flow_analysis
    
    def cluster_attention_heads(self, texts: List[str], layer: int, n_clusters: int = 3) -> Dict:
        """
        Cluster attention heads based on their attention patterns.
        
        Args:
            texts: List of texts to analyze
            layer: Layer to analyze
            n_clusters: Number of clusters to create
            
        Returns:
            Dictionary with clustering results
        """
        # Collect attention patterns for all heads across all texts
        head_patterns = []
        head_metadata = []
        
        for text_idx, text in enumerate(texts):
            attention_data = self.extractor.extract_attention_weights(text)
            
            if layer >= attention_data["num_layers"]:
                continue
                
            layer_attention = attention_data["attention_weights"][layer]
            
            for head_idx in range(layer_attention.shape[0]):
                head_attention = layer_attention[head_idx]
                
                # Flatten attention matrix as feature vector
                pattern_vector = head_attention.flatten()
                head_patterns.append(pattern_vector)
                
                head_metadata.append({
                    "text_idx": text_idx,
                    "text": text,
                    "head_idx": head_idx,
                    "layer": layer
                })
        
        if not head_patterns:
            return {"error": "No valid attention patterns found"}
        
        # Convert to numpy array
        patterns_array = np.array(head_patterns)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(50, patterns_array.shape[1]))
        patterns_reduced = pca.fit_transform(patterns_array)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(patterns_reduced)
        
        # Organize results by cluster
        clusters = {}
        for i in range(n_clusters):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_heads = [head_metadata[idx] for idx in cluster_indices]
            
            clusters[i] = {
                "heads": cluster_heads,
                "size": len(cluster_heads),
                "representative_patterns": patterns_reduced[cluster_indices]
            }
        
        return {
            "layer": layer,
            "n_clusters": n_clusters,
            "clusters": clusters,
            "pca_explained_variance": pca.explained_variance_ratio_,
            "cluster_labels": cluster_labels.tolist(),
            "cluster_centers": kmeans.cluster_centers_
        }
    
    def compare_attention_patterns(
        self, 
        texts: List[str], 
        layers: Optional[List[int]] = None
    ) -> Dict:
        """
        Compare attention patterns across different texts and layers.
        
        Args:
            texts: List of texts to compare
            layers: List of layers to analyze (if None, analyzes all layers)
            
        Returns:
            Dictionary with comparison results
        """
        if len(texts) < 2:
            raise ValueError("Need at least 2 texts for comparison")
        
        # Get attention data for all texts
        all_attention_data = []
        for text in texts:
            attention_data = self.extractor.extract_attention_weights(text)
            all_attention_data.append(attention_data)
        
        if layers is None:
            layers = list(range(min(data["num_layers"] for data in all_attention_data)))
        
        comparison_results = {
            "texts": texts,
            "layers_analyzed": layers,
            "pairwise_similarities": {},
            "layer_similarities": {},
            "attention_statistics": []
        }
        
        # Compare each pair of texts
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                text_pair = f"text_{i}_vs_text_{j}"
                similarities = []
                
                for layer in layers:
                    att1 = all_attention_data[i]["attention_weights"][layer]
                    att2 = all_attention_data[j]["attention_weights"][layer]
                    
                    # Average across heads for comparison
                    avg_att1 = np.mean(att1, axis=0)
                    avg_att2 = np.mean(att2, axis=0)
                    
                    # Ensure same dimensions (pad shorter sequence)
                    min_len = min(avg_att1.shape[0], avg_att2.shape[0])
                    avg_att1 = avg_att1[:min_len, :min_len]
                    avg_att2 = avg_att2[:min_len, :min_len]
                    
                    # Calculate similarity
                    similarity = cosine_similarity(
                        avg_att1.flatten().reshape(1, -1),
                        avg_att2.flatten().reshape(1, -1)
                    )[0, 0]
                    
                    similarities.append(float(similarity))
                
                comparison_results["pairwise_similarities"][text_pair] = {
                    "layer_similarities": similarities,
                    "average_similarity": float(np.mean(similarities))
                }
        
        # Calculate layer-wise similarities across all texts
        for layer in layers:
            layer_patterns = []
            for data in all_attention_data:
                if layer < data["num_layers"]:
                    layer_att = data["attention_weights"][layer]
                    avg_layer_att = np.mean(layer_att, axis=0)
                    layer_patterns.append(avg_layer_att.flatten())
            
            if len(layer_patterns) > 1:
                # Calculate pairwise similarities for this layer
                layer_similarities = []
                for i in range(len(layer_patterns)):
                    for j in range(i + 1, len(layer_patterns)):
                        sim = cosine_similarity(
                            np.array(layer_patterns[i]).reshape(1, -1),
                            np.array(layer_patterns[j]).reshape(1, -1)
                        )[0, 0]
                        layer_similarities.append(sim)
                
                comparison_results["layer_similarities"][layer] = {
                    "pairwise_similarities": layer_similarities,
                    "average_similarity": float(np.mean(layer_similarities))
                }
        
        # Get attention statistics for each text
        for i, text in enumerate(texts):
            stats = self.extractor.get_attention_statistics(all_attention_data[i]["attention_weights"])
            stats["text_index"] = i
            stats["text"] = text
            comparison_results["attention_statistics"].append(stats)
        
        return comparison_results
    
    def analyze_positional_attention(self, text: str, layer: int) -> Dict:
        """
        Analyze positional attention patterns (e.g., local vs global attention).
        
        Args:
            text: Input text to analyze
            layer: Layer to analyze
            
        Returns:
            Dictionary with positional attention analysis
        """
        attention_data = self.extractor.extract_attention_weights(text)
        
        if layer >= attention_data["num_layers"]:
            raise ValueError(f"Layer {layer} not available.")
        
        layer_attention = attention_data["attention_weights"][layer]
        tokens = attention_data["tokens"]
        num_heads, seq_len, _ = layer_attention.shape
        
        positional_analysis = {
            "layer": layer,
            "sequence_length": seq_len,
            "head_analyses": [],
            "aggregate_analysis": {}
        }
        
        # Analyze each head
        for head_idx in range(num_heads):
            head_attention = layer_attention[head_idx]
            
            # Calculate distance-based attention
            local_attention = 0  # Attention to adjacent tokens
            medium_attention = 0  # Attention within 3 positions
            global_attention = 0  # Attention to distant tokens
            
            for i in range(seq_len):
                for j in range(seq_len):
                    distance = abs(i - j)
                    attention_weight = head_attention[i, j]
                    
                    if distance <= 1:
                        local_attention += attention_weight
                    elif distance <= 3:
                        medium_attention += attention_weight
                    else:
                        global_attention += attention_weight
            
            # Normalize by total attention
            total_attention = local_attention + medium_attention + global_attention
            if total_attention > 0:
                local_ratio = local_attention / total_attention
                medium_ratio = medium_attention / total_attention
                global_ratio = global_attention / total_attention
            else:
                local_ratio = medium_ratio = global_ratio = 0
            
            # Calculate attention to specific positions
            attention_to_first = np.mean(head_attention[:, 0])  # Attention to first token
            attention_to_last = np.mean(head_attention[:, -1])  # Attention to last token
            
            head_analysis = {
                "head": head_idx,
                "local_attention_ratio": float(local_ratio),
                "medium_attention_ratio": float(medium_ratio),
                "global_attention_ratio": float(global_ratio),
                "attention_to_first_token": float(attention_to_first),
                "attention_to_last_token": float(attention_to_last),
                "max_attention_distance": int(np.unravel_index(np.argmax(head_attention), head_attention.shape)[0])
            }
            
            positional_analysis["head_analyses"].append(head_analysis)
        
        # Aggregate across heads
        avg_local = np.mean([h["local_attention_ratio"] for h in positional_analysis["head_analyses"]])
        avg_medium = np.mean([h["medium_attention_ratio"] for h in positional_analysis["head_analyses"]])
        avg_global = np.mean([h["global_attention_ratio"] for h in positional_analysis["head_analyses"]])
        
        positional_analysis["aggregate_analysis"] = {
            "average_local_ratio": float(avg_local),
            "average_medium_ratio": float(avg_medium),
            "average_global_ratio": float(avg_global),
            "attention_pattern_type": self._classify_attention_pattern(avg_local, avg_medium, avg_global)
        }
        
        return positional_analysis
    
    def _classify_attention_pattern(self, local_ratio: float, medium_ratio: float, global_ratio: float) -> str:
        """Classify the attention pattern based on distance ratios."""
        if local_ratio > 0.6:
            return "local"
        elif global_ratio > 0.5:
            return "global"
        elif medium_ratio > 0.4:
            return "medium-range"
        else:
            return "mixed"
    
    def generate_attention_summary(self, text: str) -> Dict:
        """
        Generate a comprehensive summary of attention patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with comprehensive attention summary
        """
        attention_data = self.extractor.extract_attention_weights(text)
        stats = self.extractor.get_attention_statistics(attention_data["attention_weights"])
        
        summary = {
            "text": text,
            "model_info": {
                "name": attention_data["model_name"],
                "num_layers": attention_data["num_layers"],
                "num_heads": attention_data["num_heads"],
                "sequence_length": attention_data["sequence_length"]
            },
            "overall_statistics": stats["overall_stats"],
            "layer_summaries": [],
            "key_findings": {}
        }
        
        # Analyze each layer
        layer_entropies = []
        layer_sparsities = []
        
        for layer_idx in range(attention_data["num_layers"]):
            layer_stats = stats["layer_stats"][layer_idx]
            
            # Positional analysis for this layer
            pos_analysis = self.analyze_positional_attention(text, layer_idx)
            
            layer_summary = {
                "layer": layer_idx,
                "statistics": layer_stats,
                "positional_pattern": pos_analysis["aggregate_analysis"]["attention_pattern_type"],
                "attention_ratios": {
                    "local": pos_analysis["aggregate_analysis"]["average_local_ratio"],
                    "medium": pos_analysis["aggregate_analysis"]["average_medium_ratio"],
                    "global": pos_analysis["aggregate_analysis"]["average_global_ratio"]
                }
            }
            
            summary["layer_summaries"].append(layer_summary)
            layer_entropies.append(layer_stats["entropy"])
            layer_sparsities.append(layer_stats["sparsity"])
        
        # Key findings
        summary["key_findings"] = {
            "highest_entropy_layer": int(np.argmax(layer_entropies)),
            "lowest_entropy_layer": int(np.argmin(layer_entropies)),
            "sparsest_layer": int(np.argmax(layer_sparsities)),
            "densest_layer": int(np.argmin(layer_sparsities)),
            "predominant_pattern": self._get_predominant_pattern(summary["layer_summaries"])
        }
        
        return summary
    
    def _get_predominant_pattern(self, layer_summaries: List[Dict]) -> str:
        """Determine the predominant attention pattern across layers."""
        patterns = [layer["positional_pattern"] for layer in layer_summaries]
        pattern_counts = {pattern: patterns.count(pattern) for pattern in set(patterns)}
        return max(pattern_counts, key=pattern_counts.get)
    
    def export_analysis_report(self, text: str, save_path: str) -> str:
        """
        Export a comprehensive analysis report to file.
        
        Args:
            text: Input text to analyze
            save_path: Path to save the report
            
        Returns:
            Path to the saved report
        """
        summary = self.generate_attention_summary(text)
        
        report_lines = [
            "# Attention Analysis Report",
            f"**Text:** {text}",
            f"**Model:** {summary['model_info']['name']}",
            f"**Generated:** {pd.Timestamp.now()}",
            "",
            "## Model Information",
            f"- Layers: {summary['model_info']['num_layers']}",
            f"- Heads per layer: {summary['model_info']['num_heads']}",
            f"- Sequence length: {summary['model_info']['sequence_length']}",
            "",
            "## Overall Statistics",
            f"- Mean attention: {summary['overall_statistics']['mean_attention']:.4f}",
            f"- Attention entropy: {summary['overall_statistics']['entropy']:.4f}",
            f"- Attention sparsity: {summary['overall_statistics']['sparsity']:.4f}",
            "",
            "## Key Findings",
            f"- Highest entropy layer: {summary['key_findings']['highest_entropy_layer']}",
            f"- Sparsest layer: {summary['key_findings']['sparsest_layer']}",
            f"- Predominant pattern: {summary['key_findings']['predominant_pattern']}",
            "",
            "## Layer-by-Layer Analysis"
        ]
        
        for layer in summary["layer_summaries"]:
            report_lines.extend([
                f"### Layer {layer['layer']}",
                f"- Pattern type: {layer['positional_pattern']}",
                f"- Entropy: {layer['statistics']['entropy']:.4f}",
                f"- Sparsity: {layer['statistics']['sparsity']:.4f}",
                f"- Local attention: {layer['attention_ratios']['local']:.2%}",
                f"- Global attention: {layer['attention_ratios']['global']:.2%}",
                ""
            ])
        
        # Write report to file
        with open(save_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return save_path 