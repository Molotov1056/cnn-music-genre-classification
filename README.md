# Music Genre Classification using CNN

A deep learning project that classifies music genres using Convolutional Neural Networks (CNN) trained on Mel-spectrograms from the GTZAN dataset.

## Features

- **CNN Architecture**: 2 Conv+Pool blocks with dropout and batch normalization
- **Audio Preprocessing**: Converts audio to Mel-spectrograms for CNN input
- **10 Genre Classification**: Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **CLI Application**: Command-line interface for batch predictions
- **Web Interface**: Streamlit app for interactive genre prediction
- **Model Persistence**: Save and load trained models

## Project Structure

```
genre_recognition/
├── requirements.txt          # Project dependencies
├── data/                    # GTZAN dataset storage
├── models/                  # Trained model storage
├── src/
│   ├── data_preprocessing.py # Audio preprocessing pipeline
│   ├── model.py             # CNN architecture definition
│   ├── train.py             # Training script
│   └── utils.py             # Helper functions
├── cli_app.py               # Command-line interface
├── streamlit_app.py         # Web interface
├── PROJECT_PLAN.md          # Project plan
└── README.md               # This file
```

## Installation

1. **Clone the repository** (if using git):
   ```bash
   git clone <repository-url>
   cd genre_recognition
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download GTZAN Dataset**:
   ```bash
   python cli_app.py download-info
   ```
   
   Follow the instructions to download and extract the dataset to `data/genres_original/`

## Usage

### 1. Training the Model

Train the CNN model on the GTZAN dataset:

```bash
python cli_app.py train
```

This will:
- Load and preprocess 1000 audio files
- Train the CNN model for up to 100 epochs
- Save the best model to `models/genre_classifier.h5`
- Generate training visualizations and test results

### 2. Command Line Prediction

Predict genre of an audio file:

```bash
python cli_app.py predict path/to/audio.wav
```

Options:
- `--model`: Path to trained model (default: `models/genre_classifier.h5`)
- `--encoder`: Path to label encoder (default: `models/label_encoder.pkl`)
- `--top-k`: Number of top predictions to show (default: 3)

Example:
```bash
python cli_app.py predict sample.wav --top-k 5
```

### 3. Web Interface

Launch the Streamlit web application:

**Option 1: Easy launcher (recommended)**
```bash
python3 launch_web_app.py
```

**Option 2: Manual launch**
```bash
streamlit run streamlit_app.py --server.port 8503
```

The app will automatically find an available port (starting from 8503) to avoid conflicts with other projects.

Features:
- Upload audio files through web interface
- Real-time genre prediction
- Confidence visualization
- Model information display
- Dataset statistics

### 4. Model Information

View trained model details:

```bash
python cli_app.py info
```

## Model Architecture

```
Input: Mel-spectrogram (128x128x1)
│
├── Conv2D(32) + BatchNorm + MaxPool2D + Dropout(0.25)
├── Conv2D(64) + BatchNorm + MaxPool2D + Dropout(0.25)
├── Flatten
├── Dense(128) + BatchNorm + Dropout(0.5)
└── Dense(10, softmax) # Output genres
```

## Audio Preprocessing Pipeline

1. **Load Audio**: Using librosa with 22.05kHz sample rate
2. **Normalize**: Amplitude normalization
3. **Duration**: Pad/truncate to 30 seconds
4. **Mel-spectrogram**: Convert to mel-spectrogram (128 mel bins)
5. **Resize**: Resize to 128x128 for CNN input

## Dataset

**GTZAN Dataset**:
- 1000 audio tracks (30 seconds each)
- 10 genres × 100 tracks per genre
- Genres: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock
- Split: 80% train, 10% validation, 10% test

## Performance

Expected performance after training:
- **Test Accuracy**: ~70-80% (typical for GTZAN dataset)
- **Training Time**: 2-4 hours (depending on hardware)
- **Model Size**: ~500KB

## Supported Audio Formats

- WAV (.wav)
- MP3 (.mp3) 
- FLAC (.flac)
- M4A (.m4a)

## Requirements

- Python 3.7+
- TensorFlow 2.12+
- librosa 0.10+
- scikit-learn 1.3+
- streamlit 1.25+
- matplotlib 3.5+
- numpy 1.21+

## Example Usage

```python
from src.utils import predict_genre

# Predict genre of an audio file
genre, confidence, top_3 = predict_genre('sample.wav')
print(f"Predicted genre: {genre} (confidence: {confidence:.2%})")
```

## Troubleshooting

1. **Module not found errors**: Make sure all dependencies are installed
2. **Audio loading errors**: Check audio file format and integrity
3. **Memory errors**: Reduce batch size in training script
4. **CUDA errors**: Ensure TensorFlow GPU setup is correct

## License

This project is for educational purposes. The GTZAN dataset has its own licensing terms.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

- GTZAN Dataset: [http://marsyas.info/downloads/datasets.html](http://marsyas.info/downloads/datasets.html)
- Original Paper: Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals.