"""
ATTRIEVAL: Attention-guided Retrieval for Long-Context Reasoning

Implementation of the ATTRIEVAL method from:
"Attention Reveals More Than Tokens: Training-Free Long-Context Reasoning with Attention-guided Retrieval"

This module provides tools for leveraging attention weights from generated Chain-of-Thought (CoT)
tokens to retrieve relevant facts from long contexts and improve reasoning performance.
"""

import torch
import numpy as np
import re
import json
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from collections import defaultdict

from .extractor import AttentionExtractor


@dataclass
class AttrievelConfig:
    """Configuration for ATTRIEVAL algorithm."""
    
    # Layer selection (use last 1/4 layers as they better ground to context)
    layer_fraction: float = 0.25
    
    # Top-k threshold for identifying highly attended tokens
    top_k: int = 50
    
    # Frequency threshold for filtering attention sinks
    frequency_threshold: float = 0.99
    
    # Minimum fact length (in tokens)
    min_fact_length: int = 3
    
    # Maximum number of facts to retrieve
    max_facts: int = 10
    
    # Number of tokens for cross-evaluation token selection
    cross_eval_tokens: int = 10


class AttrievelRetriever:
    """
    ATTRIEVAL: Attention-guided Retrieval for long-context reasoning.
    
    This class implements the core ATTRIEVAL algorithm that:
    1. Aggregates attention weights across multiple layers
    2. Segments context into facts
    3. Identifies top-k attended tokens
    4. Filters attention sinks
    5. Scores facts based on attention
    6. Optionally uses cross-evaluation for token selection
    """
    
    def __init__(
        self, 
        extractor: AttentionExtractor, 
        config: Optional[AttrievelConfig] = None
    ):
        """
        Initialize ATTRIEVAL retriever.
        
        Args:
            extractor: AttentionExtractor instance for getting attention data
            config: Configuration for ATTRIEVAL algorithm
        """
        self.extractor = extractor
        self.config = config or AttrievelConfig()
        
    def retrieve_facts(
        self, 
        context: str, 
        question: str,
        cot_response: str,
        use_cross_evaluation: bool = True
    ) -> Dict[str, Any]:
        """
        Main ATTRIEVAL algorithm for retrieving relevant facts.
        
        Args:
            context: Long input context containing facts
            question: Question to be answered
            cot_response: Generated Chain-of-Thought response
            use_cross_evaluation: Whether to use cross-evaluation for token selection
            
        Returns:
            Dictionary containing retrieved facts and analysis
        """
        
        # Step 1: Get attention weights for CoT generation
        print("🔍 Step 1: Extracting attention weights from CoT tokens...")
        attention_data = self._extract_cot_attention(context, question, cot_response)
        
        # Step 2: Aggregate attention across layers
        print("📊 Step 2: Aggregating attention across multiple layers...")
        aggregated_attention = self._aggregate_attention_weights(attention_data)
        
        # Step 3: Segment context into facts
        print("📝 Step 3: Segmenting context into discrete facts...")
        facts = self._segment_context_into_facts(context)
        
        # Step 4: Identify top-k attended tokens for each CoT token
        print("🎯 Step 4: Identifying top-k attended tokens...")
        top_k_tokens = self._identify_top_k_tokens(aggregated_attention)
        
        # Step 5: Filter attention sinks
        print("🔧 Step 5: Filtering attention sink facts...")
        filtered_facts = self._filter_attention_sinks(facts, top_k_tokens, attention_data['tokens'])
        
        # Step 6: Score facts based on attention
        print("⚖️  Step 6: Scoring facts based on attention weights...")
        fact_scores = self._score_facts(filtered_facts, aggregated_attention, attention_data['tokens'])
        
        # Step 7: Cross-evaluation token selection (optional)
        retriever_tokens = None
        if use_cross_evaluation:
            print("🔀 Step 7: Cross-evaluation token selection...")
            retriever_tokens = self._cross_evaluation_token_selection(
                context, question, cot_response
            )
            # Re-score facts using only retriever tokens
            if retriever_tokens:
                retriever_attention = aggregated_attention[retriever_tokens, :]
                fact_scores = self._score_facts(filtered_facts, retriever_attention, attention_data['tokens'])
        
        # Step 8: Select top facts
        print("🏆 Step 8: Selecting top relevant facts...")
        top_facts = self._select_top_facts(fact_scores, filtered_facts)
        
        return {
            'retrieved_facts': top_facts,
            'fact_scores': fact_scores,
            'attention_data': attention_data,
            'aggregated_attention': aggregated_attention,
            'retriever_tokens': retriever_tokens,
            'all_facts': facts,
            'filtered_facts': filtered_facts,
            'config': self.config.__dict__
        }
    
    def _extract_cot_attention(self, context: str, question: str, cot_response: str) -> Dict[str, Any]:
        """
        Extract attention weights from model when generating CoT response.
        
        For this implementation, we'll simulate by getting attention on the full prompt.
        In practice, this would require capturing attention during actual CoT generation.
        """
        full_prompt = f"{context}\n\nQuestion: {question}\n\nAnswer: {cot_response}"
        
        # Extract attention weights
        attention_data = self.extractor.extract_attention_weights(full_prompt)
        
        # Find where CoT tokens start (approximate)
        context_tokens = self.extractor.tokenizer.encode(f"{context}\n\nQuestion: {question}\n\nAnswer: ")
        cot_start_idx = len(context_tokens)
        
        attention_data['cot_start_idx'] = cot_start_idx
        attention_data['context_length'] = len(self.extractor.tokenizer.encode(context))
        
        return attention_data
    
    def _aggregate_attention_weights(self, attention_data: Dict[str, Any]) -> np.ndarray:
        """
        Aggregate attention weights across multiple layers and heads (Equation 4).
        
        Uses the last 1/4 layers as they better ground to context.
        """
        attention_weights = attention_data['attention_weights']
        num_layers = len(attention_weights)
        
        # Select last 1/4 layers
        start_layer = int(num_layers * (1 - self.config.layer_fraction))
        selected_layers = list(range(start_layer, num_layers))
        
        print(f"   📍 Using layers {selected_layers} out of {num_layers} total layers")
        
        # Aggregate across selected layers and heads
        aggregated = None
        for layer_idx in selected_layers:
            layer_attention = attention_weights[layer_idx]  # Shape: (num_heads, seq_len, seq_len)
            
            # Average across heads
            layer_avg = np.mean(layer_attention, axis=0)  # Shape: (seq_len, seq_len)
            
            if aggregated is None:
                aggregated = layer_avg
            else:
                aggregated += layer_avg
        
        # Average across layers
        aggregated = aggregated / len(selected_layers)
        
        print(f"   📏 Aggregated attention shape: {aggregated.shape}")
        return aggregated
    
    def _segment_context_into_facts(self, context: str) -> List[Dict[str, Any]]:
        """
        Segment input context into discrete facts based on punctuation.
        """
        # Split by sentences using punctuation
        sentences = re.split(r'[.!?]+', context)
        
        facts = []
        token_offset = 0
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence or len(sentence.split()) < self.config.min_fact_length:
                continue
                
            # Tokenize this fact to get token indices
            fact_tokens = self.extractor.tokenizer.encode(sentence)
            
            fact_info = {
                'id': i,
                'text': sentence,
                'token_start': token_offset,
                'token_end': token_offset + len(fact_tokens),
                'token_indices': list(range(token_offset, token_offset + len(fact_tokens))),
                'length': len(fact_tokens)
            }
            
            facts.append(fact_info)
            token_offset += len(fact_tokens)
        
        print(f"   📋 Segmented context into {len(facts)} facts")
        return facts
    
    def _identify_top_k_tokens(self, aggregated_attention: np.ndarray) -> Dict[int, List[int]]:
        """
        For each generated token, identify top-k input tokens with highest attention (Equation 5).
        """
        cot_tokens, context_tokens = aggregated_attention.shape
        top_k_tokens = {}
        
        for t in range(cot_tokens):
            # Get attention scores for this generated token
            attention_scores = aggregated_attention[t, :]
            
            # Find top-k input tokens
            top_k_indices = np.argsort(attention_scores)[-self.config.top_k:]
            top_k_tokens[t] = top_k_indices.tolist()
        
        print(f"   🎯 Identified top-{self.config.top_k} tokens for {cot_tokens} generated tokens")
        return top_k_tokens
    
    def _filter_attention_sinks(
        self, 
        facts: List[Dict[str, Any]], 
        top_k_tokens: Dict[int, List[int]],
        all_tokens: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Filter out attention sink facts that appear too frequently (Equation 6).
        """
        fact_frequencies = defaultdict(int)
        total_cot_tokens = len(top_k_tokens)
        
        # Map tokens to facts
        token_to_fact = {}
        for fact in facts:
            for token_idx in fact['token_indices']:
                if token_idx < len(all_tokens):  # Ensure within bounds
                    token_to_fact[token_idx] = fact['id']
        
        # Count how often each fact appears in top-k
        for t, top_tokens in top_k_tokens.items():
            facts_in_top_k = set()
            for token_idx in top_tokens:
                if token_idx in token_to_fact:
                    facts_in_top_k.add(token_to_fact[token_idx])
            
            for fact_id in facts_in_top_k:
                fact_frequencies[fact_id] += 1
        
        # Filter facts based on frequency threshold
        filtered_facts = []
        for fact in facts:
            frequency = fact_frequencies[fact['id']] / total_cot_tokens
            if frequency < self.config.frequency_threshold:
                fact['frequency'] = frequency
                filtered_facts.append(fact)
        
        print(f"   🔧 Filtered {len(facts) - len(filtered_facts)} attention sink facts")
        print(f"   ✅ Remaining facts: {len(filtered_facts)}")
        
        return filtered_facts
    
    def _score_facts(
        self, 
        facts: List[Dict[str, Any]], 
        attention_weights: np.ndarray,
        all_tokens: List[str]
    ) -> Dict[int, float]:
        """
        Score facts based on attention weights (Equation 7).
        """
        fact_scores = {}
        
        num_cot_tokens, num_context_tokens = attention_weights.shape
        
        for fact in facts:
            fact_id = fact['id']
            token_indices = [idx for idx in fact['token_indices'] if idx < num_context_tokens]
            
            if not token_indices:
                fact_scores[fact_id] = 0.0
                continue
            
            # Calculate average attention across all CoT tokens and fact tokens
            total_attention = 0.0
            for token_idx in token_indices:
                for cot_token in range(num_cot_tokens):
                    total_attention += attention_weights[cot_token, token_idx]
            
            # Average by number of tokens in fact and number of CoT tokens
            avg_attention = total_attention / (len(token_indices) * num_cot_tokens)
            fact_scores[fact_id] = float(avg_attention)
        
        print(f"   ⚖️  Computed scores for {len(fact_scores)} facts")
        return fact_scores
    
    def _cross_evaluation_token_selection(
        self, 
        context: str, 
        question: str, 
        cot_response: str
    ) -> Optional[List[int]]:
        """
        Cross-evaluation token selection using KL divergence (Algorithm 2).
        
        This identifies "retriever tokens" vs "reasoner tokens" by comparing
        model predictions with long vs short prompts.
        """
        try:
            # Long prompt (with context)
            long_prompt = f"{context}\n\nQuestion: {question}\n\nAnswer: {cot_response}"
            
            # Short prompt (without context)
            short_prompt = f"Question: {question}\n\nAnswer: {cot_response}"
            
            # Get model predictions for both prompts
            with torch.no_grad():
                # Tokenize both prompts
                long_inputs = self.extractor.tokenizer(long_prompt, return_tensors="pt")
                short_inputs = self.extractor.tokenizer(short_prompt, return_tensors="pt")
                
                # Get model outputs
                long_outputs = self.extractor.model(**long_inputs.to(self.extractor.device))
                short_outputs = self.extractor.model(**short_inputs.to(self.extractor.device))
                
                # Get probability distributions
                long_probs = torch.softmax(long_outputs.logits, dim=-1)
                short_probs = torch.softmax(short_outputs.logits, dim=-1)
                
                # Calculate KL divergence for each token
                # Align sequences (short prompt is subset of long prompt)
                seq_offset = long_probs.size(1) - short_probs.size(1)
                aligned_long_probs = long_probs[:, seq_offset:, :]
                
                # Compute KL divergence per token
                kl_divergences = []
                for t in range(min(aligned_long_probs.size(1), short_probs.size(1))):
                    kl_div = torch.nn.functional.kl_div(
                        torch.log(short_probs[:, t, :] + 1e-12),
                        aligned_long_probs[:, t, :],
                        reduction='sum'
                    )
                    kl_divergences.append(kl_div.item())
                
                # Select top tokens with highest KL divergence (most affected by context)
                if kl_divergences:
                    top_indices = np.argsort(kl_divergences)[-self.config.cross_eval_tokens:]
                    print(f"   🔀 Selected {len(top_indices)} retriever tokens using cross-evaluation")
                    return top_indices.tolist()
        
        except Exception as e:
            print(f"   ⚠️  Cross-evaluation failed: {e}")
            return None
    
    def _select_top_facts(
        self, 
        fact_scores: Dict[int, float], 
        facts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select top facts based on scores.
        """
        # Sort facts by score
        sorted_facts = sorted(
            facts, 
            key=lambda f: fact_scores.get(f['id'], 0.0), 
            reverse=True
        )
        
        # Take top N facts
        top_facts = sorted_facts[:self.config.max_facts]
        
        # Add scores to facts
        for fact in top_facts:
            fact['attention_score'] = fact_scores.get(fact['id'], 0.0)
        
        print(f"   🏆 Selected top {len(top_facts)} facts")
        return top_facts
    
    def visualize_retrieved_facts(self, retrieval_result: Dict[str, Any]) -> str:
        """
        Create a visualization of retrieved facts with scores.
        """
        retrieved_facts = retrieval_result['retrieved_facts']
        
        viz_text = "🎯 ATTRIEVAL: Retrieved Facts\n"
        viz_text += "=" * 50 + "\n\n"
        
        for i, fact in enumerate(retrieved_facts, 1):
            score = fact.get('attention_score', 0.0)
            frequency = fact.get('frequency', 0.0)
            
            viz_text += f"📍 Fact {i} (Score: {score:.4f}, Freq: {frequency:.3f})\n"
            viz_text += f"   {fact['text']}\n\n"
        
        viz_text += f"\n📊 Total facts retrieved: {len(retrieved_facts)}\n"
        viz_text += f"📏 Configuration: {retrieval_result['config']}\n"
        
        return viz_text
    
    def export_retrieval_result(self, retrieval_result: Dict[str, Any], filepath: str) -> None:
        """
        Export retrieval results to JSON file.
        """
        # Prepare serializable data
        export_data = {
            'retrieved_facts': retrieval_result['retrieved_facts'],
            'fact_scores': retrieval_result['fact_scores'],
            'config': retrieval_result['config'],
            'summary': {
                'total_facts_retrieved': len(retrieval_result['retrieved_facts']),
                'total_facts_analyzed': len(retrieval_result['all_facts']),
                'total_facts_filtered': len(retrieval_result['filtered_facts']),
                'used_cross_evaluation': retrieval_result['retriever_tokens'] is not None
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Exported retrieval results to {filepath}")


def create_attrieval_demo(
    context: str,
    question: str, 
    cot_response: str,
    model_name: str = "gpt2"
) -> Dict[str, Any]:
    """
    Demonstration function for ATTRIEVAL algorithm.
    
    Args:
        context: Long context containing facts
        question: Question to be answered
        cot_response: Chain-of-thought response
        model_name: Model to use for attention extraction
        
    Returns:
        Dictionary with retrieval results and analysis
    """
    from ..utils.helpers import load_model_and_tokenizer
    
    print("🚀 ATTRIEVAL Demonstration")
    print("=" * 50)
    
    # Load model and tokenizer
    print(f"📦 Loading model: {model_name}")
    model, tokenizer = load_model_and_tokenizer(model_name)
    
    # Initialize components
    extractor = AttentionExtractor(model, tokenizer)
    config = AttrievelConfig()
    retriever = AttrievelRetriever(extractor, config)
    
    # Run ATTRIEVAL
    print(f"\n🔍 Running ATTRIEVAL on context ({len(context)} chars)")
    result = retriever.retrieve_facts(context, question, cot_response)
    
    # Display results
    print("\n" + retriever.visualize_retrieved_facts(result))
    
    return result 