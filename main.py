"""
Main training script, which accepts configuration file and other training parameters from command line.
"""
import argparse
import sys
from pathlib import Path

from config import Config
import trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a segmentation model for medical images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration JSON file (e.g., config/config_atlas.json)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        required=True,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Directory path to save model checkpoints and metrics'
    )
    parser.add_argument(
        '--trainer',
        type=str,
        required=True,
        help='Trainer class name'
    )
    
    parser.add_argument(
        '--validation',
        action='store_true',
        help='Enable validation during training'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume training from last checkpoint in save_path'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    print(f"Loading configuration from: {args.config}")
    config = Config(args.config)
    
    # Create save directory if it doesn't exist
    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Model checkpoints will be saved to: {args.save_path}")
    
    TrainerClass = getattr(trainer, args.trainer)
    print(f"\nInitializing {TrainerClass.__name__}...")

    trainer_instance = TrainerClass(
        config=config,
        epochs=args.epochs,
         validation=args.validation,
        save_path=args.save_path,
        resume=args.resume,
        debug=args.debug
    )
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Validation: {'Enabled' if args.validation else 'Disabled'}")
    print(f"Resume: {'Yes' if args.resume else 'No'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    try:
        trainer_instance.train()
        print("\n" + "="*50)
        print("Training completed successfully!")
        print("="*50)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print("Checkpoint saved at last epoch.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
