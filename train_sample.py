#!/usr/bin/env python3

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pickle
from src.data_preprocessing import AudioPreprocessor
from src.model import GenreCNN

def plot_training_history(history, save_path='models/training_history.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Configuration for sample data
    DATA_PATH = 'data/genres_sample'  # Path to sample dataset
    MODEL_PATH = 'models/genre_classifier_sample.h5'
    LABEL_ENCODER_PATH = 'models/label_encoder_sample.pkl'
    
    # Check if sample data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Sample data directory {DATA_PATH} not found.")
        print("Creating sample data first...")
        
        # Create sample data
        try:
            import subprocess
            subprocess.run([sys.executable, 'create_sample_data.py'], check=True)
        except Exception as e:
            print(f"Error creating sample data: {e}")
            return
    
    # Initialize preprocessor
    print("Initializing audio preprocessor...")
    preprocessor = AudioPreprocessor()
    
    # Load and preprocess data
    print("Loading and preprocessing sample dataset...")
    X, y = preprocessor.load_gtzan_dataset(DATA_PATH)
    
    if len(X) == 0:
        print("Error: No audio files found. Please check the dataset structure.")
        return
    
    print(f"Loaded {len(X)} audio files")
    print(f"Input shape: {X[0].shape}")
    
    # Prepare data splits
    print("Preparing data splits...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder = preprocessor.prepare_data(X, y)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Save label encoder
    os.makedirs('models', exist_ok=True)
    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    # Initialize and compile model
    print("Building CNN model...")
    cnn = GenreCNN(input_shape=X_train[0].shape, num_classes=len(label_encoder.classes_))
    model = cnn.compile_model()
    
    # Print model summary
    model.summary()
    
    # Train model with fewer epochs for sample data
    print("Starting training on sample data...")
    print("Note: This is training on synthetic data for demonstration purposes.")
    history = cnn.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=8)
    
    # Plot training history
    plot_training_history(history, 'models/training_history_sample.png')
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    cnn.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save test results
    with open('models/test_results_sample.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Number of classes: {len(label_encoder.classes_)}\n")
        f.write(f"Classes: {', '.join(label_encoder.classes_)}\n")
        f.write(f"Dataset: Sample synthetic data (50 files)\n")
        f.write(f"Note: This model was trained on synthetic data for demonstration.\n")
        f.write(f"For real performance, train on the full GTZAN dataset.\n")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Model saved: {MODEL_PATH}")
    print(f"✅ Label encoder saved: {LABEL_ENCODER_PATH}")
    print(f"✅ Training history: models/training_history_sample.png")
    print(f"✅ Test results: models/test_results_sample.txt")
    print("\nYou can now test the model:")
    print(f"python3 cli_app.py predict data/genres_sample/jazz/jazz_000.wav --model {MODEL_PATH} --encoder {LABEL_ENCODER_PATH}")
    print("\nOr start the web app:")
    print("streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()