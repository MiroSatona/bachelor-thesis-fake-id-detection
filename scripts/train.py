"""
Author: Martin Zůbek
Date: 2026-04-12
File: train.py
Description: Script for training the classification head of the extractors 
for identity document forgery detection. 
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.constants import CPU, CUDA
from src.heads.constants import DROPOUT_PRECENTAGE, IMLVIT_KEY, MMFUSION_KEY, SWINT_KEY, TRUFOR_KEY
from src.model.model import Model


def parse_args() -> argparse.Namespace:
    """
    Function for creating the argument parser for the training classification 
    head of the extractors for identity document forgery detection.
    
    Arguments:
        None
    Returns:
        argparse.Namespace: Parsed arguments for training the model.
        
    """
    parser = argparse.ArgumentParser(
        description="Train the classification head for identity document forgery detection."
    )
    parser.add_argument( # Type of the feature extractor.
        "--extractor",
        type=str,
        required=True,
        choices=[TRUFOR_KEY, MMFUSION_KEY, SWINT_KEY, IMLVIT_KEY],
        help="Name of the feature extractor to use.",
    )
    parser.add_argument( # Path to the data directory for training.
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory.",
    )
    parser.add_argument( # Batch size for training.
        "--batch_size",
        type=int,
        required=True,
        help="Batch size for training.",
    )
    parser.add_argument( # Number of epochs for training.
        "--epochs",
        type=int,
        required=True,

        help="Number of training epochs.",
    )
    parser.add_argument( # Learning rate for the optimizer.
        "--lr",
        type=float,
        required=True,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument( # Dropout percentage for the MLP head.
        "--dropout",
        type=float,
        default=DROPOUT_PRECENTAGE,
        help=f"Dropout percentage for the MLP head. Default is {DROPOUT_PRECENTAGE}.",
    )
    parser.add_argument( # Device to use for training.
        "--device",
        type=str,
        choices=[CPU, CUDA],
        required=True,
        help="Device to use for training. Default is cuda if available, otherwise cpu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model: Model = Model(
        extractor_name=args.extractor,
        device= True if args.device == CUDA else False,
        data_path=args.data_path,
        batch_size=args.batch_size,
        number_of_epochs=args.epochs,
        learning_rate=args.lr,
        dropout_percentage=args.dropout,
    )
    # Run training and validation of the model
    model.train()
    # Run testing of the model
    model.test()

if __name__ == "__main__":
    main()
