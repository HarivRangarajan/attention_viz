"""Command-line interface for attention_viz."""

import argparse
import sys
import os
from typing import Optional

from .core.visualizer import AttentionVisualizer
from .core.analyzer import AttentionAnalyzer
from .utils.helpers import load_model_and_tokenizer


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Attention Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  attention-viz visualize --model gpt2 --text "Hello world" --layer 6 --head 4
  attention-viz analyze --model bert-base-uncased --text "The cat sat on the mat" --output report.md
  attention-viz compare --model gpt2 --text "Sample text" --layers 0 3 6 9
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create attention visualizations")
    viz_parser.add_argument("--model", required=True, help="Model name or path")
    viz_parser.add_argument("--text", required=True, help="Text to analyze")
    viz_parser.add_argument("--layer", type=int, help="Specific layer to visualize")
    viz_parser.add_argument("--head", type=int, help="Specific head to visualize")
    viz_parser.add_argument("--output", help="Output file path")
    viz_parser.add_argument("--interactive", action="store_true", default=True, help="Create interactive visualization")
    viz_parser.add_argument("--static", action="store_true", help="Create static visualization")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze attention patterns")
    analyze_parser.add_argument("--model", required=True, help="Model name or path")
    analyze_parser.add_argument("--text", required=True, help="Text to analyze")
    analyze_parser.add_argument("--output", help="Output report path")
    analyze_parser.add_argument("--layer", type=int, help="Specific layer to analyze")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare attention across layers")
    compare_parser.add_argument("--model", required=True, help="Model name or path")
    compare_parser.add_argument("--text", required=True, help="Text to analyze")
    compare_parser.add_argument("--layers", nargs="+", type=int, required=True, help="Layers to compare")
    compare_parser.add_argument("--output", help="Output file path")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Get attention statistics")
    stats_parser.add_argument("--model", required=True, help="Model name or path")
    stats_parser.add_argument("--text", required=True, help="Text to analyze")
    
    return parser


def handle_visualize(args) -> None:
    """Handle visualize command."""
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    print("Creating visualizer...")
    viz = AttentionVisualizer(model, tokenizer)
    
    interactive = args.interactive and not args.static
    
    print(f"Visualizing attention for: '{args.text}'")
    viz.visualize_attention(
        text=args.text,
        layer=args.layer,
        head=args.head,
        save_path=args.output,
        interactive=interactive
    )
    
    if args.output:
        print(f"✅ Visualization saved to: {args.output}")
    else:
        print("✅ Visualization displayed")


def handle_analyze(args) -> None:
    """Handle analyze command."""
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    print("Creating analyzer...")
    viz = AttentionVisualizer(model, tokenizer)
    analyzer = AttentionAnalyzer(viz.extractor)
    
    print(f"Analyzing attention for: '{args.text}'")
    
    if args.layer is not None:
        # Analyze specific layer
        pos_analysis = analyzer.analyze_positional_attention(args.text, args.layer)
        print(f"Layer {args.layer} analysis:")
        print(f"  Pattern type: {pos_analysis['aggregate_analysis']['attention_pattern_type']}")
        print(f"  Local attention: {pos_analysis['aggregate_analysis']['average_local_ratio']:.2%}")
        print(f"  Global attention: {pos_analysis['aggregate_analysis']['average_global_ratio']:.2%}")
    
    if args.output:
        # Generate full report
        report_path = analyzer.export_analysis_report(args.text, args.output)
        print(f"✅ Analysis report saved to: {report_path}")
    else:
        # Print summary to console
        summary = analyzer.generate_attention_summary(args.text)
        print("\n📊 Attention Summary:")
        print(f"  Model: {summary['model_info']['name']}")
        print(f"  Layers: {summary['model_info']['num_layers']}")
        print(f"  Heads: {summary['model_info']['num_heads']}")
        print(f"  Overall entropy: {summary['overall_statistics']['entropy']:.4f}")
        print(f"  Overall sparsity: {summary['overall_statistics']['sparsity']:.4f}")
        print(f"  Predominant pattern: {summary['key_findings']['predominant_pattern']}")


def handle_compare(args) -> None:
    """Handle compare command."""
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    print("Creating visualizer...")
    viz = AttentionVisualizer(model, tokenizer)
    
    print(f"Comparing layers {args.layers} for: '{args.text}'")
    viz.compare_layers(
        text=args.text,
        layers=args.layers,
        save_path=args.output
    )
    
    if args.output:
        print(f"✅ Layer comparison saved to: {args.output}")
    else:
        print("✅ Layer comparison displayed")


def handle_stats(args) -> None:
    """Handle stats command."""
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_and_tokenizer(args.model)
    
    print("Creating visualizer...")
    viz = AttentionVisualizer(model, tokenizer)
    
    print(f"Computing statistics for: '{args.text}'")
    stats = viz.get_attention_stats(args.text)
    
    print("\n📊 Attention Statistics:")
    print(f"  Mean attention: {stats['overall_stats']['mean_attention']:.4f}")
    print(f"  Attention entropy: {stats['overall_stats']['entropy']:.4f}")
    print(f"  Attention sparsity: {stats['overall_stats']['sparsity']:.4f}")
    print(f"  Max attention: {stats['overall_stats']['max_attention']:.4f}")
    print(f"  Min attention: {stats['overall_stats']['min_attention']:.4f}")
    
    print("\n📈 Layer-wise Statistics:")
    for layer_stat in stats['layer_stats'][:5]:  # Show first 5 layers
        print(f"  Layer {layer_stat['layer']}: entropy={layer_stat['entropy']:.4f}, sparsity={layer_stat['sparsity']:.4f}")
    
    if len(stats['layer_stats']) > 5:
        print(f"  ... and {len(stats['layer_stats']) - 5} more layers")


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "visualize":
            handle_visualize(args)
        elif args.command == "analyze":
            handle_analyze(args)
        elif args.command == "compare":
            handle_compare(args)
        elif args.command == "stats":
            handle_stats(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 