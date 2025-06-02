# Quick Start Guide

Get up and running with attention_viz in minutes!

## Installation

```bash
# Install the package
pip install -e .

# Or install with all dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Simple Attention Visualization

```python
from attention_viz import AttentionVisualizer, load_model_and_tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer("gpt2")

# Create visualizer
viz = AttentionVisualizer(model, tokenizer)

# Visualize attention for a text
text = "The cat sat on the mat"
viz.visualize_attention(text, layer=6, head=4, save_path="attention.html")
```

### 2. Compare Attention Across Layers

```python
# Compare attention patterns across multiple layers
viz.compare_layers(
    text="Hello world, how are you?",
    layers=[0, 3, 6, 9],
    save_path="layer_comparison.png"
)
```

### 3. Get Attention Statistics

```python
# Get quantitative metrics
stats = viz.get_attention_stats("Attention is all you need")
print(f"Entropy: {stats['overall_stats']['entropy']:.4f}")
print(f"Sparsity: {stats['overall_stats']['sparsity']:.4f}")
```

### 4. Advanced Analysis

```python
from attention_viz import AttentionAnalyzer

# Create analyzer
analyzer = AttentionAnalyzer(viz.extractor)

# Analyze attention patterns
pos_analysis = analyzer.analyze_positional_attention("Sample text", layer=6)
print(f"Pattern type: {pos_analysis['aggregate_analysis']['attention_pattern_type']}")

# Generate comprehensive report
analyzer.export_analysis_report("Sample text", "report.md")
```

## Examples

Run the basic example:

```bash
python examples/basic_usage.py
```

## Supported Models

- GPT-2 (gpt2, gpt2-medium, gpt2-large)
- BERT (bert-base-uncased, bert-large-uncased)
- RoBERTa (roberta-base, roberta-large)
- DistilBERT (distilbert-base-uncased)
- T5 (t5-small, t5-base)

## Key Features

✅ Interactive and static visualizations  
✅ Multi-layer attention comparison  
✅ Head specialization analysis  
✅ Attention flow analysis  
✅ Export capabilities (JSON, PNG, HTML)  
✅ Statistical analysis tools  
✅ Integration with inspectus library  

## Next Steps

- Check out the [examples directory](examples/) for more detailed tutorials
- Read the full documentation in the README
- Explore different models and texts
- Try the advanced analysis features

## Need Help?

- Check the [examples](examples/) for common use cases
- Look at the [tests](tests/) for API usage examples  
- Open an issue on GitHub for bugs or feature requests 