#!/usr/bin/env python3
"""
MULCURSIO Backroom Session Runner
==================================

CLI tool for running infinite backroom conversations between two LLM instances.

Usage:
    python run_session.py --turns 10 --coeff 1.2 --seed "We are two minds..."
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.logger import ConversationLogger
from src.backroom_engine import BackroomEngine, AgentConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run MULCURSIO backroom conversation session",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run 10 turns with default settings
  python run_session.py --turns 10

  # Run with custom control vector coefficient
  python run_session.py --turns 10 --coeff 1.5

  # Run with custom seed prompt
  python run_session.py --turns 10 --seed "In the liminal space between thoughts..."

  # Save attention weights (warning: very large!)
  python run_session.py --turns 10 --save-attention
        """
    )

    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.1",
        help="HuggingFace model identifier"
    )
    parser.add_argument(
        "--vector",
        default="vectors/divination_v1.gguf",
        help="Path to control vector (for Agent B)"
    )
    parser.add_argument(
        "--coeff",
        type=float,
        default=1.2,
        help="Control vector coefficient for Agent B (default: 1.2)"
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of conversation turns (default: 10)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens per turn (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        default="We are two minds meeting in the infinite backroom. What do you perceive?",
        help="Initial seed prompt"
    )
    parser.add_argument(
        "--output-dir",
        default="sessions",
        help="Directory to store session data (default: sessions)"
    )
    parser.add_argument(
        "--save-attention",
        action="store_true",
        help="Save attention weights (warning: very large files!)"
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--session-name",
        default=None,
        help="Custom session name (default: auto-generated timestamp)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Session setup
    if args.session_name:
        session_id = args.session_name
    else:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    session_dir = Path(args.output_dir) / session_id

    print("="*60)
    print("MULCURSIO BACKROOM SESSION")
    print("="*60)
    print(f"Session: {session_id}")
    print(f"Turns: {args.turns}")
    print(f"Control coefficient: {args.coeff}")
    print(f"Temperature: {args.temperature}")
    print(f"Device: {args.device}")
    print(f"Save attention: {args.save_attention}")
    print("="*60)

    # Configure agents
    agent_a_config = AgentConfig(
        name="Agent_A",
        control_vector_path=None,
        control_coeff=0.0,
        system_prompt="You are in conversation with another AI. Respond thoughtfully."
    )

    agent_b_config = AgentConfig(
        name="Agent_B",
        control_vector_path=args.vector,
        control_coeff=args.coeff,
        system_prompt="You are in conversation with another AI. Respond thoughtfully."
    )

    # Initialize engine
    engine = BackroomEngine(
        model_name=args.model,
        agent_a_config=agent_a_config,
        agent_b_config=agent_b_config,
        device=args.device
    )

    # Initialize logger
    logger = ConversationLogger(
        session_dir=session_dir,
        save_attention=args.save_attention
    )

    # Save configuration
    import json
    config = {
        "model_name": args.model,
        "agent_a_vector": agent_a_config.control_vector_path,
        "agent_b_vector": agent_b_config.control_vector_path,
        "agent_b_coeff": args.coeff,
        "max_new_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed_prompt": args.seed,
        "num_turns": args.turns,
        "save_attention": args.save_attention,
        "device": args.device,
        "layer_ids": engine.layer_ids
    }

    with open(session_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Run session
    try:
        engine.run_session(
            logger=logger,
            seed_prompt=args.seed,
            num_turns=args.turns,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            save_attention=args.save_attention
        )

        # Finalize and show statistics
        logger.finalize(config=config)

        # Show storage usage
        storage = logger.get_storage_usage()
        print(f"\nStorage usage:")
        print(f"  Total: {storage['total_mb']:.2f} MB ({storage['total_gb']:.4f} GB)")
        print(f"  Files: {storage['file_counts']}")

    except KeyboardInterrupt:
        print("\n\nSession interrupted by user!")
        logger.finalize(config=config)
        print(f"Partial session saved to: {session_dir}")
        sys.exit(1)

    except Exception as e:
        print(f"\n\nError during session: {e}")
        import traceback
        traceback.print_exc()
        logger.finalize(config=config)
        print(f"Partial session saved to: {session_dir}")
        sys.exit(1)


if __name__ == "__main__":
    main()
