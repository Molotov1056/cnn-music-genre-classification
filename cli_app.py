#!/usr/bin/env python3

import sys
import os
import argparse
from src.utils import predict_genre, validate_audio_file, get_model_info, download_sample_data

def main():
    parser = argparse.ArgumentParser(description='Music Genre Classification CLI')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict genre of audio file')
    predict_parser.add_argument('audio_file', help='Path to audio file')
    predict_parser.add_argument('--model', default='models/genre_classifier.h5', 
                               help='Path to trained model (default: models/genre_classifier.h5)')
    predict_parser.add_argument('--encoder', default='models/label_encoder.pkl',
                               help='Path to label encoder (default: models/label_encoder.pkl)')
    predict_parser.add_argument('--top-k', type=int, default=3,
                               help='Show top K predictions (default: 3)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show model information')
    info_parser.add_argument('--model', default='models/genre_classifier.h5',
                            help='Path to trained model (default: models/genre_classifier.h5)')
    
    # Download command
    download_parser = subparsers.add_parser('download-info', help='Show dataset download instructions')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        predict_command(args)
    elif args.command == 'info':
        info_command(args)
    elif args.command == 'download-info':
        download_info_command()
    elif args.command == 'train':
        train_command()
    else:
        parser.print_help()

def predict_command(args):
    print(f"Analyzing audio file: {args.audio_file}")
    
    # Validate audio file
    is_valid, message = validate_audio_file(args.audio_file)
    if not is_valid:
        print(f"Error: {message}")
        sys.exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Please train the model first using: python cli_app.py train")
        sys.exit(1)
    
    if not os.path.exists(args.encoder):
        print(f"Error: Label encoder not found: {args.encoder}")
        print("Please train the model first using: python cli_app.py train")
        sys.exit(1)
    
    try:
        # Make prediction
        predicted_genre, confidence, top_predictions = predict_genre(
            args.audio_file, args.model, args.encoder
        )
        
        if predicted_genre is None:
            print("Error: Could not process audio file")
            sys.exit(1)
        
        # Display results
        print("\n" + "="*50)
        print("GENRE PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Genre: {predicted_genre.upper()}")
        print(f"Confidence: {confidence:.2%}")
        print("\n" + "-"*30)
        print(f"Top {min(args.top_k, len(top_predictions))} Predictions:")
        print("-"*30)
        
        for i, (genre, conf) in enumerate(top_predictions[:args.top_k], 1):
            print(f"{i}. {genre.upper():<12} {conf:.2%}")
        
        print("="*50)
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

def info_command(args):
    print("Model Information")
    print("="*30)
    
    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        return
    
    try:
        info = get_model_info(args.model)
        if info:
            print(f"Input Shape: {info['input_shape']}")
            print(f"Output Shape: {info['output_shape']}")
            print(f"Total Parameters: {info['total_params']:,}")
            print(f"Number of Layers: {info['layers']}")
        
        # Check for test results
        test_results_path = 'models/test_results.txt'
        if os.path.exists(test_results_path):
            print("\nTest Results:")
            print("-" * 20)
            with open(test_results_path, 'r') as f:
                print(f.read())
        
    except Exception as e:
        print(f"Error reading model info: {e}")

def download_info_command():
    print("GTZAN Dataset Download Instructions")
    print("="*40)
    instructions = download_sample_data()
    print(instructions)

def train_command():
    print("Starting model training...")
    print("This may take several hours depending on your hardware.")
    
    # Check if dataset exists
    data_path = 'data/genres_original'
    if not os.path.exists(data_path):
        print(f"\nError: Dataset not found at {data_path}")
        print("Please download the GTZAN dataset first:")
        print("python cli_app.py download-info")
        sys.exit(1)
    
    # Import and run training
    try:
        from src.train import main as train_main
        train_main()
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Please make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    main()