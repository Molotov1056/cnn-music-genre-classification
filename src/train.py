import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from data_preprocessing import AudioPreprocessor
from model import GenreCNN

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
    # Configuration
    DATA_PATH = 'data/genres_original'  # Path to GTZAN dataset
    MODEL_PATH = 'models/genre_classifier.h5'
    LABEL_ENCODER_PATH = 'models/label_encoder.pkl'
    
    # Check if data directory exists
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data directory {DATA_PATH} not found.")
        print("Please download the GTZAN dataset and extract it to the data/ folder.")
        print("The folder structure should be: data/genres_original/[genre]/[audio_files].wav")
        return
    
    # Initialize preprocessor
    print("Initializing audio preprocessor...")
    preprocessor = AudioPreprocessor()
    
    # Load and preprocess data
    print("Loading and preprocessing GTZAN dataset...")
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
    
    # Train model
    print("Starting training...")
    history = cnn.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = cnn.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    cnn.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    # Save test results
    with open('models/test_results.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Number of classes: {len(label_encoder.classes_)}\n")
        f.write(f"Classes: {', '.join(label_encoder.classes_)}\n")

if __name__ == "__main__":
    main()