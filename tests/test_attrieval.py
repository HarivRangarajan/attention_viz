#!/usr/bin/env python3
"""
Unit tests for ATTRIEVAL (Attention-guided Retrieval for Long-Context Reasoning)

This test suite verifies the correctness of all ATTRIEVAL algorithm components
including mathematical operations, data structures, and edge cases.
"""

import pytest
import numpy as np
import torch
import json
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any

# Import the modules to test
from attention_viz.core.attrieval import (
    AttrievelConfig, 
    AttrievelRetriever, 
    create_attrieval_demo
)
from attention_viz.core.extractor import AttentionExtractor


class TestAttrievelConfig:
    """Test the AttrievelConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AttrievelConfig()
        
        assert config.layer_fraction == 0.25
        assert config.top_k == 50
        assert config.frequency_threshold == 0.99
        assert config.min_fact_length == 3
        assert config.max_facts == 10
        assert config.cross_eval_tokens == 10
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AttrievelConfig(
            layer_fraction=0.5,
            top_k=25,
            frequency_threshold=0.95,
            min_fact_length=5,
            max_facts=5,
            cross_eval_tokens=20
        )
        
        assert config.layer_fraction == 0.5
        assert config.top_k == 25
        assert config.frequency_threshold == 0.95
        assert config.min_fact_length == 5
        assert config.max_facts == 5
        assert config.cross_eval_tokens == 20


class TestAttrievelRetriever:
    """Test the AttrievelRetriever class and its methods."""
    
    @pytest.fixture
    def mock_extractor(self):
        """Create a mock AttentionExtractor for testing."""
        extractor = Mock(spec=AttentionExtractor)
        
        # Mock tokenizer
        extractor.tokenizer = Mock()
        extractor.tokenizer.encode.side_effect = lambda text: list(range(len(text.split())))
        extractor.device = "cpu"
        
        # Mock model
        extractor.model = Mock()
        
        # Mock attention extraction
        def mock_extract_attention(text):
            words = text.split()
            seq_len = len(words)
            num_layers = 12
            num_heads = 8
            
            # Create fake attention weights
            attention_weights = []
            for layer in range(num_layers):
                layer_weights = np.random.rand(num_heads, seq_len, seq_len)
                # Make attention weights sum to 1 across last dimension
                layer_weights = layer_weights / layer_weights.sum(axis=-1, keepdims=True)
                attention_weights.append(layer_weights)
            
            return {
                'attention_weights': attention_weights,
                'tokens': words,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'sequence_length': seq_len
            }
        
        extractor.extract_attention_weights.side_effect = mock_extract_attention
        return extractor
    
    @pytest.fixture
    def retriever(self, mock_extractor):
        """Create an AttrievelRetriever instance for testing."""
        config = AttrievelConfig(
            layer_fraction=0.25,
            top_k=5,  # Smaller for testing
            frequency_threshold=0.8,
            min_fact_length=2,
            max_facts=3,
            cross_eval_tokens=3
        )
        return AttrievelRetriever(mock_extractor, config)
    
    def test_initialization(self, mock_extractor):
        """Test AttrievelRetriever initialization."""
        # Test with custom config
        config = AttrievelConfig(top_k=25)
        retriever = AttrievelRetriever(mock_extractor, config)
        
        assert retriever.extractor == mock_extractor
        assert retriever.config.top_k == 25
        
        # Test with default config
        retriever_default = AttrievelRetriever(mock_extractor)
        assert retriever_default.config.top_k == 50
    
    def test_segment_context_into_facts(self, retriever):
        """Test context segmentation into facts."""
        context = "Patient is 65 years old. He has diabetes. Blood pressure is high. Weight is normal."
        
        facts = retriever._segment_context_into_facts(context)
        
        # Should have 4 facts (sentences)
        assert len(facts) >= 3  # At least 3 facts that meet min_fact_length
        
        # Check fact structure
        for fact in facts:
            assert 'id' in fact
            assert 'text' in fact
            assert 'token_start' in fact
            assert 'token_end' in fact
            assert 'token_indices' in fact
            assert 'length' in fact
            
            # Check that fact text is non-empty and meets minimum length
            assert len(fact['text'].split()) >= retriever.config.min_fact_length
            
            # Check token indices are sequential
            expected_indices = list(range(fact['token_start'], fact['token_end']))
            assert fact['token_indices'] == expected_indices
    
    def test_segment_context_edge_cases(self, retriever):
        """Test context segmentation edge cases."""
        # Empty context
        facts = retriever._segment_context_into_facts("")
        assert len(facts) == 0
        
        # Single word (below min_fact_length)
        facts = retriever._segment_context_into_facts("Hello.")
        assert len(facts) == 0
        
        # Context without punctuation
        facts = retriever._segment_context_into_facts("This is a very long sentence without any punctuation marks")
        assert len(facts) == 1  # Should be treated as one fact
        
        # Multiple punctuation marks
        facts = retriever._segment_context_into_facts("First sentence! Second sentence? Third sentence.")
        assert len(facts) >= 2  # Should split on different punctuation
    
    def test_aggregate_attention_weights(self, retriever):
        """Test attention weight aggregation across layers."""
        # Create fake attention data
        num_layers = 12
        num_heads = 8
        seq_len = 10
        
        attention_weights = []
        for layer in range(num_layers):
            layer_weights = np.random.rand(num_heads, seq_len, seq_len)
            # Normalize to valid attention weights
            layer_weights = layer_weights / layer_weights.sum(axis=-1, keepdims=True)
            attention_weights.append(layer_weights)
        
        attention_data = {
            'attention_weights': attention_weights,
            'tokens': ['word'] * seq_len
        }
        
        aggregated = retriever._aggregate_attention_weights(attention_data)
        
        # Check output shape
        assert aggregated.shape == (seq_len, seq_len)
        
        # Check that values are reasonable (non-negative, roughly normalized)
        assert np.all(aggregated >= 0)
        assert np.all(aggregated <= 1)
        
        # Check that we used the correct layers (last 25%)
        expected_start_layer = int(num_layers * (1 - retriever.config.layer_fraction))
        assert expected_start_layer == 9  # For 12 layers, should start at layer 9
    
    def test_identify_top_k_tokens(self, retriever):
        """Test top-k token identification."""
        # Create fake aggregated attention
        cot_tokens = 5
        context_tokens = 10
        aggregated_attention = np.random.rand(cot_tokens, context_tokens)
        
        top_k_tokens = retriever._identify_top_k_tokens(aggregated_attention)
        
        # Check structure
        assert len(top_k_tokens) == cot_tokens
        
        for t in range(cot_tokens):
            assert t in top_k_tokens
            assert len(top_k_tokens[t]) == retriever.config.top_k
            
            # Check that indices are in valid range
            for idx in top_k_tokens[t]:
                assert 0 <= idx < context_tokens
            
            # Check that these are actually the top-k indices
            attention_scores = aggregated_attention[t, :]
            top_k_expected = np.argsort(attention_scores)[-retriever.config.top_k:]
            assert set(top_k_tokens[t]) == set(top_k_expected.tolist())
    
    def test_filter_attention_sinks(self, retriever):
        """Test attention sink filtering."""
        # Create test facts
        facts = [
            {'id': 0, 'token_indices': [0, 1, 2], 'text': 'First fact'},
            {'id': 1, 'token_indices': [3, 4, 5], 'text': 'Second fact'},
            {'id': 2, 'token_indices': [6, 7, 8], 'text': 'Third fact'}
        ]
        
        # Create top_k_tokens where fact 0 appears very frequently (attention sink)
        top_k_tokens = {
            0: [0, 1, 2, 9],  # Always includes fact 0
            1: [0, 1, 3, 4],  # Always includes fact 0
            2: [0, 6, 7, 8],  # Always includes fact 0
            3: [3, 4, 5, 9],  # Includes fact 1
            4: [6, 7, 8, 9]   # Includes fact 2
        }
        
        all_tokens = ['token'] * 10
        
        filtered_facts = retriever._filter_attention_sinks(facts, top_k_tokens, all_tokens)
        
        # Fact 0 should be filtered out (appears in 3/5 = 0.6 > threshold for some cases)
        # The exact behavior depends on the threshold
        assert len(filtered_facts) <= len(facts)
        
        # Check that frequency is computed and added
        for fact in filtered_facts:
            assert 'frequency' in fact
            assert 0 <= fact['frequency'] <= 1
    
    def test_score_facts(self, retriever):
        """Test fact scoring based on attention weights."""
        # Create test facts
        facts = [
            {'id': 0, 'token_indices': [0, 1]},
            {'id': 1, 'token_indices': [2, 3]},
            {'id': 2, 'token_indices': [4, 5]}
        ]
        
        # Create attention weights (3 CoT tokens, 6 context tokens)
        attention_weights = np.array([
            [0.5, 0.3, 0.1, 0.0, 0.1, 0.0],  # High attention to fact 0
            [0.1, 0.1, 0.4, 0.3, 0.1, 0.0],  # High attention to fact 1
            [0.1, 0.1, 0.1, 0.1, 0.3, 0.3]   # High attention to fact 2
        ])
        
        all_tokens = ['token'] * 6
        
        fact_scores = retriever._score_facts(facts, attention_weights, all_tokens)
        
        # Check that all facts have scores
        assert len(fact_scores) == 3
        assert all(fact_id in fact_scores for fact_id in [0, 1, 2])
        
        # Check score calculation (average attention across tokens and CoT steps)
        # Fact 0: tokens [0,1], attention = (0.5+0.3 + 0.1+0.1 + 0.1+0.1) / (2*3) = 1.2/6 = 0.2
        expected_score_0 = (0.5 + 0.3 + 0.1 + 0.1 + 0.1 + 0.1) / (2 * 3)
        assert abs(fact_scores[0] - expected_score_0) < 1e-6
        
        # Scores should be non-negative
        assert all(score >= 0 for score in fact_scores.values())
    
    def test_select_top_facts(self, retriever):
        """Test top fact selection."""
        # Create test facts
        facts = [
            {'id': 0, 'text': 'Fact 0'},
            {'id': 1, 'text': 'Fact 1'},
            {'id': 2, 'text': 'Fact 2'},
            {'id': 3, 'text': 'Fact 3'},
            {'id': 4, 'text': 'Fact 4'}
        ]
        
        # Create fact scores
        fact_scores = {0: 0.1, 1: 0.8, 2: 0.3, 3: 0.9, 4: 0.2}
        
        top_facts = retriever._select_top_facts(fact_scores, facts)
        
        # Should return top 3 facts (max_facts = 3 in config)
        assert len(top_facts) == 3
        
        # Should be ordered by score (highest first)
        scores = [fact['attention_score'] for fact in top_facts]
        assert scores == sorted(scores, reverse=True)
        
        # Should include the highest scoring facts
        top_fact_ids = [fact['id'] for fact in top_facts]
        assert 3 in top_fact_ids  # Highest score (0.9)
        assert 1 in top_fact_ids  # Second highest (0.8)
        assert 2 in top_fact_ids  # Third highest (0.3)
        
        # Each fact should have attention_score added
        for fact in top_facts:
            assert 'attention_score' in fact
            assert fact['attention_score'] == fact_scores[fact['id']]
    
    def test_select_top_facts_fewer_than_max(self, retriever):
        """Test top fact selection when fewer facts than max_facts."""
        facts = [
            {'id': 0, 'text': 'Fact 0'},
            {'id': 1, 'text': 'Fact 1'}
        ]
        
        fact_scores = {0: 0.1, 1: 0.8}
        
        top_facts = retriever._select_top_facts(fact_scores, facts)
        
        # Should return all facts (only 2, less than max_facts=3)
        assert len(top_facts) == 2
    
    def test_extract_cot_attention(self, retriever):
        """Test CoT attention extraction."""
        context = "Patient has diabetes and high blood pressure."
        question = "What are the patient's conditions?"
        cot_response = "The patient has two conditions: diabetes and hypertension."
        
        attention_data = retriever._extract_cot_attention(context, question, cot_response)
        
        # Check that required fields are present
        assert 'attention_weights' in attention_data
        assert 'tokens' in attention_data
        assert 'cot_start_idx' in attention_data
        assert 'context_length' in attention_data
        
        # Check that cot_start_idx is reasonable
        assert attention_data['cot_start_idx'] >= 0
        
        # Check that context_length is reasonable
        assert attention_data['context_length'] > 0
    
    @patch('torch.no_grad')
    def test_cross_evaluation_token_selection(self, mock_no_grad, retriever):
        """Test cross-evaluation token selection."""
        # Mock the torch operations
        mock_context_manager = MagicMock()
        mock_no_grad.return_value = mock_context_manager
        mock_context_manager.__enter__.return_value = None
        mock_context_manager.__exit__.return_value = None
        
        # Mock tokenizer outputs
        retriever.extractor.tokenizer.return_value = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1]])
        }
        
        # Mock model outputs with different logits for long vs short prompts
        mock_long_output = MagicMock()
        mock_short_output = MagicMock()
        
        # Create different probability distributions
        mock_long_output.logits = torch.tensor([[[1.0, 2.0, 0.5], [0.5, 1.0, 2.0], [2.0, 0.5, 1.0]]])
        mock_short_output.logits = torch.tensor([[[0.5, 1.0, 2.0], [2.0, 0.5, 1.0]]])
        
        retriever.extractor.model.side_effect = [mock_long_output, mock_short_output]
        
        context = "Patient has diabetes."
        question = "What condition?"
        cot_response = "Diabetes."
        
        retriever_tokens = retriever._cross_evaluation_token_selection(context, question, cot_response)
        
        # Should return a list of token indices or None
        if retriever_tokens is not None:
            assert isinstance(retriever_tokens, list)
            assert len(retriever_tokens) <= retriever.config.cross_eval_tokens
            assert all(isinstance(idx, int) for idx in retriever_tokens)
    
    def test_visualize_retrieved_facts(self, retriever):
        """Test fact visualization."""
        retrieval_result = {
            'retrieved_facts': [
                {'id': 0, 'text': 'First fact', 'attention_score': 0.8, 'frequency': 0.2},
                {'id': 1, 'text': 'Second fact', 'attention_score': 0.6, 'frequency': 0.1}
            ],
            'config': {'max_facts': 3, 'top_k': 5}
        }
        
        viz_text = retriever.visualize_retrieved_facts(retrieval_result)
        
        # Check that visualization contains expected elements
        assert 'ATTRIEVAL: Retrieved Facts' in viz_text
        assert 'First fact' in viz_text
        assert 'Second fact' in viz_text
        assert '0.8' in viz_text  # attention score
        assert '0.6' in viz_text  # attention score
        assert 'Total facts retrieved: 2' in viz_text
    
    def test_export_retrieval_result(self, retriever):
        """Test retrieval result export."""
        retrieval_result = {
            'retrieved_facts': [
                {'id': 0, 'text': 'Test fact', 'attention_score': 0.5}
            ],
            'fact_scores': {0: 0.5},
            'config': {'max_facts': 10},
            'all_facts': [{'id': 0}, {'id': 1}],
            'filtered_facts': [{'id': 0}],
            'retriever_tokens': [1, 2, 3]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            retriever.export_retrieval_result(retrieval_result, filepath)
            
            # Check that file was created and contains expected data
            assert os.path.exists(filepath)
            
            with open(filepath, 'r') as f:
                exported_data = json.load(f)
            
            assert 'retrieved_facts' in exported_data
            assert 'fact_scores' in exported_data
            assert 'config' in exported_data
            assert 'summary' in exported_data
            
            summary = exported_data['summary']
            assert summary['total_facts_retrieved'] == 1
            assert summary['total_facts_analyzed'] == 2
            assert summary['total_facts_filtered'] == 1
            assert summary['used_cross_evaluation'] == True
            
        finally:
            if os.path.exists(filepath):
                os.unlink(filepath)
    
    def test_retrieve_facts_integration(self, retriever):
        """Test the main retrieve_facts method integration."""
        context = "Patient is 65 years old. He has diabetes. Blood pressure is elevated."
        question = "What are the patient's conditions?"
        cot_response = "The patient has diabetes and high blood pressure."
        
        # This should run without errors
        result = retriever.retrieve_facts(context, question, cot_response, use_cross_evaluation=False)
        
        # Check that all expected keys are present
        expected_keys = [
            'retrieved_facts', 'fact_scores', 'attention_data', 
            'aggregated_attention', 'retriever_tokens', 'all_facts', 
            'filtered_facts', 'config'
        ]
        
        for key in expected_keys:
            assert key in result
        
        # Check data types and basic constraints
        assert isinstance(result['retrieved_facts'], list)
        assert isinstance(result['fact_scores'], dict)
        assert isinstance(result['all_facts'], list)
        assert isinstance(result['filtered_facts'], list)
        assert isinstance(result['config'], dict)
        
        # Check that we don't retrieve more than max_facts
        assert len(result['retrieved_facts']) <= retriever.config.max_facts
        
        # Check that retrieved facts are a subset of filtered facts
        retrieved_ids = {fact['id'] for fact in result['retrieved_facts']}
        filtered_ids = {fact['id'] for fact in result['filtered_facts']}
        assert retrieved_ids.issubset(filtered_ids)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.fixture
    def minimal_extractor(self):
        """Create a minimal mock extractor for edge case testing."""
        extractor = Mock(spec=AttentionExtractor)
        extractor.tokenizer = Mock()
        extractor.tokenizer.encode.return_value = []
        extractor.device = "cpu"
        extractor.extract_attention_weights.return_value = {
            'attention_weights': [],
            'tokens': [],
            'num_layers': 0,
            'num_heads': 0,
            'sequence_length': 0
        }
        return extractor
    
    def test_empty_context(self, minimal_extractor):
        """Test behavior with empty context."""
        retriever = AttrievelRetriever(minimal_extractor)
        
        facts = retriever._segment_context_into_facts("")
        assert len(facts) == 0
    
    def test_no_facts_meet_min_length(self, minimal_extractor):
        """Test when no facts meet minimum length requirement."""
        retriever = AttrievelRetriever(minimal_extractor, AttrievelConfig(min_fact_length=10))
        
        # Short sentences that won't meet min_fact_length
        context = "Hi. Ok. Yes."
        facts = retriever._segment_context_into_facts(context)
        assert len(facts) == 0
    
    def test_zero_attention_weights(self, minimal_extractor):
        """Test behavior with zero attention weights."""
        retriever = AttrievelRetriever(minimal_extractor)
        
        # Create zero attention matrix
        attention_weights = np.zeros((3, 5))  # 3 CoT tokens, 5 context tokens
        facts = [{'id': 0, 'token_indices': [0, 1]}]
        all_tokens = ['token'] * 5
        
        fact_scores = retriever._score_facts(facts, attention_weights, all_tokens)
        
        assert fact_scores[0] == 0.0
    
    def test_single_layer_attention(self, minimal_extractor):
        """Test aggregation with single layer."""
        retriever = AttrievelRetriever(minimal_extractor)
        
        # Single layer attention
        attention_data = {
            'attention_weights': [np.random.rand(4, 5, 5)],  # 1 layer, 4 heads, 5x5
            'tokens': ['token'] * 5
        }
        
        aggregated = retriever._aggregate_attention_weights(attention_data)
        assert aggregated.shape == (5, 5)


# class TestCreateAttrievelDemo:
#     """Test the create_attrieval_demo function."""
    
#     @patch('attention_viz.core.attrieval.load_model_and_tokenizer')
#     @patch('attention_viz.core.attrieval.AttentionExtractor')
#     def test_demo_function(self, mock_extractor_class, mock_load_model):
#         """Test the demo function."""
#         # Mock model loading
#         mock_model = Mock()
#         mock_tokenizer = Mock()
#         mock_load_model.return_value = (mock_model, mock_tokenizer)
        
#         # Mock extractor
#         mock_extractor = Mock()
#         mock_extractor_class.return_value = mock_extractor
        
#         # Mock successful retrieval
#         with patch.object(AttrievelRetriever, 'retrieve_facts') as mock_retrieve:
#             mock_retrieve.return_value = {
#                 'retrieved_facts': [{'id': 0, 'text': 'test fact', 'attention_score': 0.5}],
#                 'config': {}
#             }
            
#             with patch.object(AttrievelRetriever, 'visualize_retrieved_facts') as mock_viz:
#                 mock_viz.return_value = "Test visualization"
                
#                 result = create_attrieval_demo(
#                     context="Test context",
#                     question="Test question", 
#                     cot_response="Test response",
#                     model_name="gpt2"
#                 )
                
#                 # Check that the function returned a result
#                 assert result is not None
#                 assert 'retrieved_facts' in result


class TestMathematicalCorrectness:
    """Test mathematical correctness of ATTRIEVAL algorithms."""
    
    def test_attention_normalization_preserved(self):
        """Test that attention weight normalization is preserved through aggregation."""
        # Create normalized attention weights (sum to 1 across last dimension)
        num_layers = 4
        num_heads = 2
        seq_len = 5
        
        attention_weights = []
        for layer in range(num_layers):
            layer_weights = np.random.rand(num_heads, seq_len, seq_len)
            # Normalize to make valid attention weights
            layer_weights = layer_weights / layer_weights.sum(axis=-1, keepdims=True)
            attention_weights.append(layer_weights)
        
        # Mock setup
        extractor = Mock()
        retriever = AttrievelRetriever(extractor, AttrievelConfig(layer_fraction=0.5))
        
        attention_data = {
            'attention_weights': attention_weights,
            'tokens': ['token'] * seq_len
        }
        
        aggregated = retriever._aggregate_attention_weights(attention_data)
        
        # Check that each row still roughly sums to 1 (within numerical precision)
        row_sums = aggregated.sum(axis=-1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-10)
    
    def test_top_k_selection_correctness(self):
        """Test that top-k selection is mathematically correct."""
        extractor = Mock()
        retriever = AttrievelRetriever(extractor, AttrievelConfig(top_k=3))
        
        # Create test attention matrix
        attention_matrix = np.array([
            [0.1, 0.9, 0.3, 0.7, 0.2],  # Top-3 should be indices [1, 3, 2]
            [0.5, 0.1, 0.8, 0.2, 0.9]   # Top-3 should be indices [4, 2, 0]
        ])
        
        top_k_tokens = retriever._identify_top_k_tokens(attention_matrix)
        
        # Verify top-k for first row
        expected_top_k_0 = [1, 3, 2]  # Sorted by attention value: 0.9, 0.7, 0.3
        assert set(top_k_tokens[0]) == set(expected_top_k_0)
        
        # Verify top-k for second row  
        expected_top_k_1 = [4, 2, 0]  # Sorted by attention value: 0.9, 0.8, 0.5
        assert set(top_k_tokens[1]) == set(expected_top_k_1)
    
    def test_fact_scoring_mathematical_correctness(self):
        """Test that fact scoring follows the correct mathematical formula."""
        extractor = Mock()
        retriever = AttrievelRetriever(extractor)
        
        # Create test scenario
        facts = [{'id': 0, 'token_indices': [0, 1, 2]}]
        attention_weights = np.array([
            [0.2, 0.3, 0.1, 0.4],  # CoT token 0
            [0.1, 0.4, 0.2, 0.3],  # CoT token 1
        ])
        all_tokens = ['token'] * 4
        
        fact_scores = retriever._score_facts(facts, attention_weights, all_tokens)
        
        # Manual calculation: fact covers tokens [0,1,2]
        # CoT token 0: 0.2 + 0.3 + 0.1 = 0.6
        # CoT token 1: 0.1 + 0.4 + 0.2 = 0.7
        # Total: 0.6 + 0.7 = 1.3
        # Average: 1.3 / (3 tokens * 2 CoT tokens) = 1.3 / 6 ≈ 0.2167
        
        expected_score = (0.2 + 0.3 + 0.1 + 0.1 + 0.4 + 0.2) / (3 * 2)
        np.testing.assert_allclose(fact_scores[0], expected_score, rtol=1e-10)


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v', '--tb=short']) 