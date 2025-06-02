"""Basic tests for attention_viz library."""

import unittest
import numpy as np
from unittest.mock import Mock, patch
import torch

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_viz.core.extractor import AttentionExtractor
from attention_viz.core.analyzer import AttentionAnalyzer
from attention_viz.utils.helpers import (
    validate_attention_data, 
    get_attention_summary_stats,
    calculate_attention_distance_matrix
)


class TestAttentionExtractor(unittest.TestCase):
    """Test the AttentionExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock model and tokenizer
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        
        # Configure mock tokenizer
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "offset_mapping": [(0, 3), (4, 9), (10, 15), (16, 19)]
        }
        
        # Configure mock model outputs
        mock_attention = torch.randn(1, 12, 4, 4)  # (batch, heads, seq_len, seq_len)
        mock_outputs = Mock()
        mock_outputs.attentions = [mock_attention]  # Single layer for testing
        
        self.mock_model.return_value = mock_outputs
        self.mock_model.config.name_or_path = "test-model"
        
        # Initialize extractor
        with patch('attention_viz.core.extractor.torch.cuda.is_available', return_value=False):
            self.extractor = AttentionExtractor(self.mock_model, self.mock_tokenizer)
    
    def test_extract_attention_weights(self):
        """Test basic attention weight extraction."""
        result = self.extractor.extract_attention_weights("test text")
        
        # Check result structure
        self.assertIn("attention_weights", result)
        self.assertIn("tokens", result)
        self.assertIn("num_layers", result)
        self.assertIn("num_heads", result)
        self.assertIn("sequence_length", result)
        
        # Check dimensions
        self.assertEqual(result["num_layers"], 1)
        self.assertEqual(result["num_heads"], 12)
        self.assertEqual(result["sequence_length"], 4)
        self.assertEqual(len(result["attention_weights"]), 1)
        self.assertEqual(result["attention_weights"][0].shape, (12, 4, 4))
    
    def test_get_attention_statistics(self):
        """Test attention statistics computation."""
        # Create sample attention weights
        attention_weights = [np.random.rand(12, 4, 4)]
        
        stats = self.extractor.get_attention_statistics(attention_weights)
        
        # Check structure
        self.assertIn("layer_stats", stats)
        self.assertIn("overall_stats", stats)
        self.assertEqual(len(stats["layer_stats"]), 1)
        
        # Check layer stats
        layer_stat = stats["layer_stats"][0]
        self.assertIn("layer", layer_stat)
        self.assertIn("mean_attention", layer_stat)
        self.assertIn("entropy", layer_stat)
        self.assertIn("sparsity", layer_stat)


class TestAttentionAnalyzer(unittest.TestCase):
    """Test the AttentionAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock extractor
        self.mock_extractor = Mock()
        
        # Configure mock attention data
        attention_data = {
            "attention_weights": [np.random.rand(12, 6, 6)],  # 1 layer, 12 heads, 6x6 tokens
            "tokens": ["The", "quick", "brown", "fox", "jumps", "over"],
            "num_layers": 1,
            "num_heads": 12,
            "sequence_length": 6
        }
        
        self.mock_extractor.extract_attention_weights.return_value = attention_data
        self.mock_extractor._compute_entropy = Mock(return_value=2.5)
        self.mock_extractor._compute_sparsity = Mock(return_value=0.3)
        
        self.analyzer = AttentionAnalyzer(self.mock_extractor)
    
    def test_analyze_attention_flow(self):
        """Test attention flow analysis."""
        result = self.analyzer.analyze_attention_flow("test text", layer=0)
        
        # Check structure
        self.assertIn("layer", result)
        self.assertIn("tokens", result)
        self.assertIn("head_flows", result)
        self.assertIn("aggregate_flows", result)
        
        # Check that we analyzed all heads
        self.assertEqual(len(result["head_flows"]), 12)
        
        # Check head flow structure
        head_flow = result["head_flows"][0]
        self.assertIn("head", head_flow)
        self.assertIn("outgoing_attention", head_flow)
        self.assertIn("incoming_attention", head_flow)
        self.assertIn("attention_hubs", head_flow)
    
    def test_analyze_positional_attention(self):
        """Test positional attention analysis."""
        result = self.analyzer.analyze_positional_attention("test text", layer=0)
        
        # Check structure
        self.assertIn("layer", result)
        self.assertIn("sequence_length", result)
        self.assertIn("head_analyses", result)
        self.assertIn("aggregate_analysis", result)
        
        # Check aggregate analysis
        aggregate = result["aggregate_analysis"]
        self.assertIn("average_local_ratio", aggregate)
        self.assertIn("average_global_ratio", aggregate)
        self.assertIn("attention_pattern_type", aggregate)


class TestHelperFunctions(unittest.TestCase):
    """Test helper utility functions."""
    
    def test_validate_attention_data(self):
        """Test attention data validation."""
        # Valid data
        valid_data = {
            "attention_weights": [np.random.rand(12, 4, 4)],
            "tokens": ["a", "b", "c", "d"],
            "num_layers": 1,
            "num_heads": 12,
            "sequence_length": 4
        }
        
        self.assertTrue(validate_attention_data(valid_data))
        
        # Invalid data - missing key
        invalid_data = valid_data.copy()
        del invalid_data["tokens"]
        
        self.assertFalse(validate_attention_data(invalid_data))
    
    def test_get_attention_summary_stats(self):
        """Test attention summary statistics."""
        attention_weights = [
            np.random.rand(12, 4, 4),
            np.random.rand(12, 4, 4)
        ]
        
        stats = get_attention_summary_stats(attention_weights)
        
        # Check required statistics
        required_stats = ["mean", "std", "min", "max", "median", "q25", "q75"]
        for stat in required_stats:
            self.assertIn(stat, stats)
            self.assertIsInstance(stats[stat], float)
    
    def test_calculate_attention_distance_matrix(self):
        """Test attention distance matrix calculation."""
        attention_matrix = np.random.rand(5, 5)
        distance_matrix = calculate_attention_distance_matrix(attention_matrix)
        
        # Check shape
        self.assertEqual(distance_matrix.shape, (5, 5))
        
        # Check diagonal is zeros
        np.testing.assert_array_equal(np.diag(distance_matrix), np.zeros(5))
        
        # Check symmetry
        np.testing.assert_array_equal(distance_matrix, distance_matrix.T)


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    @patch('attention_viz.utils.helpers.AutoTokenizer')
    @patch('attention_viz.utils.helpers.AutoModel')
    def test_load_model_and_tokenizer(self, mock_model, mock_tokenizer):
        """Test model and tokenizer loading."""
        from attention_viz.utils.helpers import load_model_and_tokenizer
        
        # Configure mocks
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        with patch('torch.cuda.is_available', return_value=False):
            model, tokenizer = load_model_and_tokenizer("test-model")
        
        # Check that methods were called
        mock_tokenizer.from_pretrained.assert_called_once()
        mock_model.from_pretrained.assert_called_once()
        
        self.assertIsNotNone(model)
        self.assertIsNotNone(tokenizer)


if __name__ == "__main__":
    unittest.main() 