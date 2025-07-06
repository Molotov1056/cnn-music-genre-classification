import os
import pickle
import numpy as np
import tensorflow as tf
from .data_preprocessing import AudioPreprocessor

def load_trained_model(model_path='models/genre_classifier.h5'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    return model

def load_label_encoder(encoder_path='models/label_encoder.pkl'):
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
    
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    return label_encoder

def predict_genre(audio_file_path, model_path='models/genre_classifier.h5', 
                 encoder_path='models/label_encoder.pkl'):
    # Load model and label encoder
    model = load_trained_model(model_path)
    label_encoder = load_label_encoder(encoder_path)
    
    # Preprocess audio
    preprocessor = AudioPreprocessor()
    mel_spec = preprocessor.preprocess_audio(audio_file_path)
    
    if mel_spec is None:
        return None, None, None
    
    # Reshape for prediction
    mel_spec_batch = np.expand_dims(mel_spec, axis=0)
    
    # Make prediction
    predictions = model.predict(mel_spec_batch, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    # Convert to genre name
    predicted_genre = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    # Get top 3 predictions
    top_3_indices = np.argsort(predictions[0])[::-1][:3]
    top_3_genres = [(label_encoder.inverse_transform([idx])[0], predictions[0][idx]) 
                   for idx in top_3_indices]
    
    return predicted_genre, confidence, top_3_genres

def get_model_info(model_path='models/genre_classifier.h5'):
    if not os.path.exists(model_path):
        return None
    
    model = load_trained_model(model_path)
    
    info = {
        'input_shape': model.input.shape,
        'output_shape': model.output.shape,
        'total_params': model.count_params(),
        'layers': len(model.layers)
    }
    
    return info

def validate_audio_file(file_path):
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not file_path.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
        return False, "Unsupported audio format. Supported formats: .wav, .mp3, .flac, .m4a"
    
    return True, "Valid audio file"

def download_sample_data():
    """
    Instructions for downloading GTZAN dataset
    """
    instructions = """
    To download the GTZAN dataset:
    
    1. Visit: http://marsyas.info/downloads/datasets.html
    2. Download 'genres.tar.gz' (approximately 1.2GB)
    3. Extract the archive to get 'genres' folder
    4. Rename 'genres' to 'genres_original'
    5. Move the 'genres_original' folder to the 'data/' directory
    
    The final structure should be:
    data/
    └── genres_original/
        ├── blues/
        ├── classical/
        ├── country/
        ├── disco/
        ├── hiphop/
        ├── jazz/
        ├── metal/
        ├── pop/
        ├── reggae/
        └── rock/
    
    Each genre folder contains 100 audio files (30 seconds each).
    """
    return instructions