#!/usr/bin/env python3
"""
Simple ATTRIEVAL Test

A quick test to verify the ATTRIEVAL implementation is working correctly.
This is a simplified version of the full demo for testing purposes.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from attention_viz import (
    AttentionExtractor, 
    AttrievelRetriever, 
    AttrievelConfig,
    load_model_and_tokenizer
)


def simple_test():
    """
    Simple test of ATTRIEVAL with a minimal example.
    """
    
    print("🧪 Simple ATTRIEVAL Test")
    print("=" * 30)
    
    # Simple context with clear facts
    context = """
    The price of apples is 10 dollars.
    There are many fruits in the store.
    The price of bananas is 5 dollars less than apples.
    Customers love fresh produce.
    """
    
    question = "What is the price of bananas?"
    cot_response = "To find banana price, I need apple price first. Bananas cost 5 less than apples."
    
    print(f"Context: {context.strip()}")
    print(f"Question: {question}")
    print(f"CoT: {cot_response}")
    
    try:
        # Load a small model for testing
        print("\n📦 Loading model...")
        model, tokenizer = load_model_and_tokenizer("gpt2")
        
        # Initialize ATTRIEVAL
        print("🔧 Initializing ATTRIEVAL...")
        extractor = AttentionExtractor(model, tokenizer)
        config = AttrievelConfig(max_facts=5, top_k=20)
        retriever = AttrievelRetriever(extractor, config)
        
        # Run ATTRIEVAL
        print("🔍 Running ATTRIEVAL...")
        result = retriever.retrieve_facts(
            context=context,
            question=question,
            cot_response=cot_response,
            use_cross_evaluation=False  # Skip for simplicity
        )
        
        # Display results
        print("\n🎯 Results:")
        print(f"Retrieved {len(result['retrieved_facts'])} facts:")
        
        for i, fact in enumerate(result['retrieved_facts'], 1):
            score = fact.get('attention_score', 0.0)
            print(f"  {i}. [{score:.4f}] {fact['text'].strip()}")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1) 