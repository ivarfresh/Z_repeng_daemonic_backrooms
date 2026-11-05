#!/usr/bin/env python3
"""
Export Session Data for Visualization
======================================

Prepare session data for use in visualization systems.
Creates simplified, aggregated JSON exports suitable for video rendering.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_session_summary(session_dir: Path, output_path: Path):
    """
    Export high-level session summary as JSON.

    Args:
        session_dir: Path to session directory
        output_path: Where to save summary JSON
    """
    # Load manifest
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Load vocab
    vocab_path = session_dir / "vocab.json"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # Collect aggregated metrics
    turns_data = []

    for turn_info in manifest['turns']:
        turn_id = turn_info['turn_id']
        agent = turn_info['agent']
        metrics_file = session_dir / turn_info['metrics_file']

        if not metrics_file.exists():
            continue

        metrics = np.load(metrics_file)

        # Aggregate metrics for this turn
        turn_data = {
            'turn_id': turn_id,
            'agent': agent,
            'timestamp': turn_info['timestamp'],
            'control_coeff': turn_info['control_coeff'],
            'token_count': turn_info['token_count'],
            'response': turn_info['response'],

            # Aggregated metrics
            'entropy': {
                'mean': float(metrics['entropy'].mean()),
                'max': float(metrics['entropy'].max()),
                'min': float(metrics['entropy'].min()),
                'std': float(metrics['entropy'].std())
            },
            'confidence': {
                'mean': float(metrics['chosen_prob'].mean()),
                'max': float(metrics['chosen_prob'].max()),
                'min': float(metrics['chosen_prob'].min()),
                'std': float(metrics['chosen_prob'].std())
            },
            'chosen_rank': {
                'mean': float(metrics['chosen_rank'].mean()),
                'max': int(metrics['chosen_rank'].max()),
                'min': int(metrics['chosen_rank'].min())
            }
        }

        turns_data.append(turn_data)

    # Create summary
    summary = {
        'session_id': manifest['session_id'],
        'created_at': manifest['created_at'],
        'total_turns': manifest['total_turns'],
        'config': manifest['config'],
        'turns': turns_data
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary exported to: {output_path}")
    print(f"Total turns: {len(turns_data)}")


def export_turn_details(session_dir: Path, turn_id: int, output_path: Path, top_k: int = 5):
    """
    Export detailed token-level data for a specific turn.

    Args:
        session_dir: Path to session directory
        turn_id: Which turn to export
        output_path: Where to save detail JSON
        top_k: Number of top alternatives to include per token
    """
    # Load manifest
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    if turn_id >= len(manifest['turns']):
        print(f"Error: Turn {turn_id} not found (session has {len(manifest['turns'])} turns)")
        return

    turn_info = manifest['turns'][turn_id]
    metrics_file = session_dir / turn_info['metrics_file']

    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        return

    # Load vocab
    vocab_path = session_dir / "vocab.json"
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # Load metrics
    metrics = np.load(metrics_file)

    # Decode tokens
    output_ids = metrics['output_ids']
    tokens = [vocab.get(str(int(tid)), "<UNK>") for tid in output_ids]

    # Build token-level details
    tokens_data = []

    for i in range(len(output_ids)):
        token_data = {
            'position': i,
            'token_id': int(output_ids[i]),
            'token': tokens[i],
            'entropy': float(metrics['entropy'][i]),
            'chosen_prob': float(metrics['chosen_prob'][i]),
            'chosen_rank': int(metrics['chosen_rank'][i]),
            'alternatives': []
        }

        # Add top-k alternatives
        for k in range(min(top_k, metrics['top_k_tokens'].shape[1])):
            alt_id = int(metrics['top_k_tokens'][i, k])
            alt_token = vocab.get(str(alt_id), "<UNK>")
            alt_prob = float(metrics['top_k_probs'][i, k])

            token_data['alternatives'].append({
                'rank': k,
                'token_id': alt_id,
                'token': alt_token,
                'prob': alt_prob
            })

        tokens_data.append(token_data)

    # Create detail object
    detail = {
        'session_id': manifest['session_id'],
        'turn_id': turn_id,
        'agent': turn_info['agent'],
        'timestamp': turn_info['timestamp'],
        'control_coeff': turn_info['control_coeff'],
        'prompt': turn_info['prompt'],
        'response': turn_info['response'],
        'token_count': len(tokens),
        'tokens': tokens_data
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(detail, f, indent=2)

    print(f"Turn {turn_id} details exported to: {output_path}")
    print(f"Tokens: {len(tokens)}")


def export_highlights(session_dir: Path, output_path: Path, num_highlights: int = 10):
    """
    Export the most interesting moments from a session.

    Args:
        session_dir: Path to session directory
        output_path: Where to save highlights JSON
        num_highlights: Number of highlight moments to extract
    """
    # Load manifest
    manifest_path = session_dir / "manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Collect metrics for all turns
    turns_metrics = []

    for turn_info in manifest['turns']:
        metrics_file = session_dir / turn_info['metrics_file']

        if not metrics_file.exists():
            continue

        metrics = np.load(metrics_file)

        turns_metrics.append({
            'turn_id': turn_info['turn_id'],
            'agent': turn_info['agent'],
            'response': turn_info['response'],
            'mean_entropy': float(metrics['entropy'].mean()),
            'max_entropy': float(metrics['entropy'].max()),
            'mean_confidence': float(metrics['chosen_prob'].mean()),
            'min_confidence': float(metrics['chosen_prob'].min()),
            'entropy_variance': float(metrics['entropy'].var())
        })

    # Find highlights based on different criteria
    highlights = {
        'highest_entropy': sorted(turns_metrics, key=lambda x: x['mean_entropy'], reverse=True)[:num_highlights],
        'lowest_confidence': sorted(turns_metrics, key=lambda x: x['mean_confidence'])[:num_highlights],
        'highest_entropy_variance': sorted(turns_metrics, key=lambda x: x['entropy_variance'], reverse=True)[:num_highlights],
    }

    # Create highlights object
    highlights_data = {
        'session_id': manifest['session_id'],
        'num_highlights': num_highlights,
        'highlights': highlights
    }

    # Save
    with open(output_path, 'w') as f:
        json.dump(highlights_data, f, indent=2)

    print(f"Highlights exported to: {output_path}")
    print(f"Categories: {list(highlights.keys())}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Export session data for visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export session summary
  python export_for_viz.py summary ../sessions/session_20260130_143022

  # Export specific turn details
  python export_for_viz.py turn ../sessions/session_20260130_143022 --turn-id 42

  # Export highlights
  python export_for_viz.py highlights ../sessions/session_20260130_143022 --num-highlights 20
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Export command')

    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Export session summary')
    summary_parser.add_argument('session_dir', type=Path, help='Session directory')
    summary_parser.add_argument('--output', '-o', type=Path, default=None,
                                help='Output path (default: session_dir/summary.json)')

    # Turn details command
    turn_parser = subparsers.add_parser('turn', help='Export turn details')
    turn_parser.add_argument('session_dir', type=Path, help='Session directory')
    turn_parser.add_argument('--turn-id', type=int, required=True, help='Turn ID to export')
    turn_parser.add_argument('--top-k', type=int, default=5,
                            help='Number of alternatives to include (default: 5)')
    turn_parser.add_argument('--output', '-o', type=Path, default=None,
                            help='Output path (default: session_dir/turn_XXXX_detail.json)')

    # Highlights command
    highlights_parser = subparsers.add_parser('highlights', help='Export highlight moments')
    highlights_parser.add_argument('session_dir', type=Path, help='Session directory')
    highlights_parser.add_argument('--num-highlights', type=int, default=10,
                                   help='Number of highlights per category (default: 10)')
    highlights_parser.add_argument('--output', '-o', type=Path, default=None,
                                   help='Output path (default: session_dir/highlights.json)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == 'summary':
        output = args.output or (args.session_dir / 'summary.json')
        export_session_summary(args.session_dir, output)

    elif args.command == 'turn':
        output = args.output or (args.session_dir / f'turn_{args.turn_id:04d}_detail.json')
        export_turn_details(args.session_dir, args.turn_id, output, args.top_k)

    elif args.command == 'highlights':
        output = args.output or (args.session_dir / 'highlights.json')
        export_highlights(args.session_dir, output, args.num_highlights)


if __name__ == "__main__":
    main()
