import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

class AudioPreprocessor:
    def __init__(self, sample_rate=22050, duration=30, n_mels=128, hop_length=512):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.target_length = sample_rate * duration
        
    def load_audio(self, file_path):
        try:
            audio, _ = librosa.load(file_path, sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def normalize_audio(self, audio):
        return librosa.util.normalize(audio)
    
    def pad_or_truncate(self, audio):
        if len(audio) > self.target_length:
            return audio[:self.target_length]
        else:
            return np.pad(audio, (0, self.target_length - len(audio)), 'constant')
    
    def audio_to_melspectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def preprocess_audio(self, file_path):
        audio = self.load_audio(file_path)
        if audio is None:
            return None
        
        audio = self.normalize_audio(audio)
        audio = self.pad_or_truncate(audio)
        mel_spec = self.audio_to_melspectrogram(audio)
        
        # Resize to fixed dimensions for CNN input
        mel_spec_resized = tf.image.resize(
            np.expand_dims(mel_spec, axis=-1),
            [128, 128]
        ).numpy()
        
        return mel_spec_resized
    
    def load_gtzan_dataset(self, data_path):
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        X = []
        y = []
        
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if not os.path.exists(genre_path):
                print(f"Warning: Genre folder {genre_path} not found")
                continue
                
            for filename in os.listdir(genre_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_path, filename)
                    mel_spec = self.preprocess_audio(file_path)
                    
                    if mel_spec is not None:
                        X.append(mel_spec)
                        y.append(genre)
                        
                        if len(X) % 50 == 0:
                            print(f"Processed {len(X)} files...")
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.5):
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        y_categorical = tf.keras.utils.to_categorical(y_encoded)
        
        # Split data: 80% train, 10% validation, 10% test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_encoder