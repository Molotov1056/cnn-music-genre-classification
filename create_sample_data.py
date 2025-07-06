#!/usr/bin/env python3

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

def create_sample_audio_data():
    """
    Create sample audio files for testing the genre classification system.
    This generates synthetic audio samples that mimic different genres.
    """
    
    # Create data directory structure
    base_path = 'data/genres_sample'
    genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    
    for genre in genres:
        os.makedirs(os.path.join(base_path, genre), exist_ok=True)
    
    # Parameters
    sample_rate = 22050
    duration = 30  # seconds
    samples = sample_rate * duration
    
    print("Creating sample audio files...")
    
    for genre in genres:
        genre_path = os.path.join(base_path, genre)
        
        # Create 5 sample files per genre for testing
        for i in range(5):
            filename = f"{genre}_{i:03d}.wav"
            filepath = os.path.join(genre_path, filename)
            
            # Generate different types of synthetic audio for each genre
            audio = generate_genre_audio(genre, samples, sample_rate)
            
            # Normalize audio
            audio = audio / np.max(np.abs(audio))
            
            # Save as WAV file
            sf.write(filepath, audio, sample_rate)
            
            print(f"Created: {filepath}")
    
    print(f"\nSample dataset created in {base_path}/")
    print("This is a small synthetic dataset for testing purposes.")
    print("For real training, download the GTZAN dataset using:")
    print("python3 cli_app.py download-info")

def generate_genre_audio(genre, samples, sample_rate):
    """
    Generate synthetic audio that loosely mimics different genres.
    This is for testing purposes only.
    """
    t = np.linspace(0, samples/sample_rate, samples)
    
    if genre == 'classical':
        # Classical: Complex harmonics, orchestral-like
        audio = (np.sin(2 * np.pi * 440 * t) + 
                0.5 * np.sin(2 * np.pi * 880 * t) + 
                0.3 * np.sin(2 * np.pi * 1320 * t))
        # Add some vibrato
        vibrato = 1 + 0.1 * np.sin(2 * np.pi * 5 * t)
        audio = audio * vibrato
        
    elif genre == 'rock':
        # Rock: Distorted, power chords
        audio = np.sin(2 * np.pi * 220 * t) + 0.8 * np.sin(2 * np.pi * 330 * t)
        # Add distortion
        audio = np.tanh(3 * audio)
        
    elif genre == 'jazz':
        # Jazz: Complex chords, swing rhythm
        audio = (np.sin(2 * np.pi * 261.63 * t) +  # C
                0.6 * np.sin(2 * np.pi * 329.63 * t) +  # E
                0.4 * np.sin(2 * np.pi * 415.30 * t))   # G#
        # Add swing pattern
        swing = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)
        audio = audio * swing
        
    elif genre == 'disco':
        # Disco: Four-on-the-floor, funky bass
        kick_pattern = signal.square(2 * np.pi * 2 * t) * 0.5
        bass = np.sin(2 * np.pi * 110 * t) * 0.8
        audio = kick_pattern + bass
        
    elif genre == 'hiphop':
        # Hip-hop: Strong beat, lower frequencies
        beat = signal.square(2 * np.pi * 1.33 * t) * 0.6
        bass = np.sin(2 * np.pi * 80 * t) * 0.9
        audio = beat + bass
        
    elif genre == 'metal':
        # Metal: Heavy distortion, power chords
        audio = (np.sin(2 * np.pi * 146.83 * t) +  # D
                np.sin(2 * np.pi * 196.00 * t))     # G
        # Heavy distortion
        audio = np.tanh(5 * audio)
        
    elif genre == 'blues':
        # Blues: 12-bar progression, bends
        audio = np.sin(2 * np.pi * 146.83 * t)  # D
        # Add blues bends
        bend = 1 + 0.05 * np.sin(2 * np.pi * 0.5 * t)
        audio = np.sin(2 * np.pi * 146.83 * t * bend)
        
    elif genre == 'country':
        # Country: Twangy, major scales
        audio = (np.sin(2 * np.pi * 261.63 * t) +  # C
                0.7 * np.sin(2 * np.pi * 329.63 * t) +  # E
                0.5 * np.sin(2 * np.pi * 392.00 * t))   # G
        
    elif genre == 'pop':
        # Pop: Catchy melody, 4/4 time
        melody = np.sin(2 * np.pi * 523.25 * t)  # C5
        rhythm = 1 + 0.2 * signal.square(2 * np.pi * 2 * t)
        audio = melody * rhythm
        
    elif genre == 'reggae':
        # Reggae: Off-beat emphasis, skank
        skank = signal.square(2 * np.pi * 2 * t + np.pi/2) * 0.3
        bass = np.sin(2 * np.pi * 110 * t) * 0.8
        audio = skank + bass
    
    else:
        # Default: Simple sine wave
        audio = np.sin(2 * np.pi * 440 * t)
    
    # Add some noise for realism
    noise = np.random.normal(0, 0.05, samples)
    audio = audio + noise
    
    return audio

if __name__ == "__main__":
    create_sample_audio_data()