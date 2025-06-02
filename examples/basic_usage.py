#!/usr/bin/env python3
"""
Basic usage example for attention_viz library.

This script demonstrates how to use the attention visualization library
to analyze attention patterns in transformer models.
"""

import os
import sys

# Add the parent directory to path to import attention_viz
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_viz import AttentionVisualizer, AttentionAnalyzer, load_model_and_tokenizer


def main():
    """Demonstrate basic usage of attention_viz library."""
    
    print("🔍 Attention Viz - Basic Usage Example")
    print("=" * 50)
    
    # 1. Load model and tokenizer
    print("\n1. Loading GPT-2 model and tokenizer...")
    model_name = "gpt2"
    model, tokenizer = load_model_and_tokenizer(model_name)
    print(f"✅ Loaded {model_name}")
    
    # 2. Initialize visualizer
    print("\n2. Initializing attention visualizer...")
    viz = AttentionVisualizer(model, tokenizer)
    print("✅ Visualizer initialized")
    
    # 3. Sample text for analysis
    sample_text = "The quick brown fox jumps over the lazy dog"
    print(f"\n3. Analyzing text: '{sample_text}'")
    
    # 4. Basic attention visualization
    print("\n4. Creating basic attention visualization...")
    viz.visualize_attention(
        text=sample_text,
        layer=6,  # Middle layer
        head=4,   # Specific head
        save_path="examples/outputs/basic_attention.html",
        interactive=True
    )
    print("✅ Basic visualization saved to examples/outputs/basic_attention.html")
    
    # 5. Static heatmap
    print("\n5. Creating attention heatmap...")
    viz.plot_attention_heatmap(
        text=sample_text,
        layer=6,
        head=4,
        title="GPT-2 Attention Pattern",
        save_path="examples/outputs/attention_heatmap.png"
    )
    print("✅ Heatmap saved to examples/outputs/attention_heatmap.png")
    
    # 6. Compare multiple layers
    print("\n6. Comparing attention across layers...")
    viz.compare_layers(
        text=sample_text,
        layers=[0, 3, 6, 9],  # Early, middle, and late layers
        save_path="examples/outputs/layer_comparison.png"
    )
    print("✅ Layer comparison saved to examples/outputs/layer_comparison.png")
    
    # 7. Get attention statistics
    print("\n7. Computing attention statistics...")
    stats = viz.get_attention_stats(sample_text)
    print(f"   📊 Overall attention entropy: {stats['overall_stats']['entropy']:.4f}")
    print(f"   📊 Overall attention sparsity: {stats['overall_stats']['sparsity']:.4f}")
    print(f"   📊 Max attention weight: {stats['overall_stats']['max_attention']:.4f}")
    
    # 8. Advanced analysis with AttentionAnalyzer
    print("\n8. Running advanced analysis...")
    analyzer = AttentionAnalyzer(viz.extractor)
    
    # Positional attention analysis
    pos_analysis = analyzer.analyze_positional_attention(sample_text, layer=6)
    pattern_type = pos_analysis["aggregate_analysis"]["attention_pattern_type"]
    local_ratio = pos_analysis["aggregate_analysis"]["average_local_ratio"]
    global_ratio = pos_analysis["aggregate_analysis"]["average_global_ratio"]
    
    print(f"   🔍 Attention pattern type: {pattern_type}")
    print(f"   🔍 Local attention ratio: {local_ratio:.2%}")
    print(f"   🔍 Global attention ratio: {global_ratio:.2%}")
    
    # 9. Export attention data
    print("\n9. Exporting attention data...")
    export_path = viz.export_attention_data(
        text=sample_text,
        format="json",
        save_path="examples/outputs/attention_data.json"
    )
    print(f"✅ Attention data exported to {export_path}")
    
    # 10. Generate comprehensive report
    print("\n10. Generating analysis report...")
    report_path = analyzer.export_analysis_report(
        text=sample_text,
        save_path="examples/outputs/attention_report.md"
    )
    print(f"✅ Analysis report saved to {report_path}")
    
    # 11. Head specialization analysis (if you have multiple texts)
    print("\n11. Analyzing head specialization...")
    sample_texts = [
        "The cat sat on the mat",
        "Machine learning is fascinating",
        "Attention mechanisms are powerful",
        "Natural language processing rocks"
    ]
    
    head_analysis = viz.analyze_head_specialization(
        texts=sample_texts,
        layer=6
    )
    
    viz.plot_head_specialization(
        head_analysis,
        save_path="examples/outputs/head_specialization.png"
    )
    print("✅ Head specialization analysis saved")
    
    # 12. Using inspectus visualization (if available)
    print("\n12. Creating inspectus visualizations...")
    try:
        viz.use_inspectus_visualization(
            text=sample_text,
            save_dir="examples/outputs/inspectus_plots"
        )
        print("✅ Inspectus visualizations saved to examples/outputs/inspectus_plots/")
    except ImportError:
        print("⚠️  Inspectus not available. Install with: pip install inspectus")
    
    print("\n🎉 Example completed! Check the examples/outputs/ directory for results.")
    print("\nNext steps:")
    print("- Try different models (bert-base-uncased, roberta-base, etc.)")
    print("- Experiment with different texts")
    print("- Explore different layers and heads")
    print("- Use the interactive visualizations")


def create_output_directory():
    """Create output directory for examples."""
    output_dir = "examples/outputs"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


if __name__ == "__main__":
    # Create output directory
    create_output_directory()
    
    # Run the example
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        import traceback
        traceback.print_exc() 