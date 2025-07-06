# CNN Music Genre Classification Project Plan

## Overview
Build a CNN in TensorFlow/Keras to classify music genres using Mel-spectrograms from the GTZAN dataset, with both CLI and Streamlit interfaces.

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
└── PROJECT_PLAN.md          # This plan document

```

## Implementation Steps

### Phase 1: Environment Setup & Data Acquisition
1. **Dependencies**: Install TensorFlow, Keras, librosa, scikit-learn, numpy, matplotlib, streamlit
2. **GTZAN Dataset**: Download and extract 1000 audio tracks (10 genres × 100 tracks)

### Phase 2: Data Preprocessing Pipeline
1. **Audio Loading**: Use librosa to load .wav files
2. **Normalization**: Normalize audio amplitude
3. **Duration Handling**: Pad/truncate to fixed length (30 seconds)
4. **Mel-spectrogram**: Convert to mel-spectrograms and resize for CNN input
5. **Data Splitting**: 80% train, 10% validation, 10% test

### Phase 3: CNN Model Development
1. **Architecture**: 
   - 2 Conv2D + MaxPooling2D blocks
   - Flatten layer
   - Dense layer with dropout
   - Softmax output (10 classes)
2. **Compilation**: Adam optimizer, categorical crossentropy loss
3. **Training**: With validation monitoring and model checkpointing

### Phase 4: Application Development
1. **CLI App**: Load audio → preprocess → predict → display result
2. **Streamlit App**: Web interface with file upload and prediction visualization

### Phase 5: Testing & Validation
1. Test both applications with sample audio files
2. Validate model performance on test set

## Expected Deliverables
- Trained CNN model for genre classification
- CLI application for batch prediction
- Streamlit web application for interactive use
- Complete preprocessing pipeline
- Project documentation

## Current Progress (2025-07-06)

### ✅ Completed
- **Environment Setup**: All dependencies installed (TensorFlow, librosa, scikit-learn, streamlit)
- **Project Structure**: Complete folder structure with src/ package
- **Data Preprocessing**: Full audio preprocessing pipeline implemented
  - Audio loading with librosa
  - Normalization and padding/truncation
  - Mel-spectrogram conversion
  - Data splitting (80/10/10)
- **CNN Architecture**: Complete model implementation
  - 2 Conv2D + MaxPooling blocks
  - Batch normalization and dropout
  - Dense layers with softmax output
- **Training Pipeline**: Full training script with validation monitoring
- **CLI Application**: Command-line interface with prediction capabilities
- **Streamlit App**: Web interface with file upload and visualization
- **Documentation**: README and project plan
- **GitHub Repository**: https://github.com/Molotov1056/cnn-music-genre-classification

### 🔄 Next Steps
1. **Download GTZAN Dataset**: Users need to download the 1.2GB dataset
2. **Model Training**: Run training on the dataset (2-4 hours)
3. **Testing**: Validate both CLI and web applications
4. **Performance Evaluation**: Test accuracy and create visualizations

### 📋 Usage Instructions
```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset instructions
python3 cli_app.py download-info

# Train model (after dataset download)
python3 cli_app.py train

# CLI prediction
python3 cli_app.py predict audio.wav

# Web interface
streamlit run streamlit_app.py
```

### 🎯 Project Status: 85% Complete
- Core implementation: 100% ✅
- Dataset acquisition: Pending user action 🔄
- Model training: Ready to execute 🔄
- Testing: Pending 🔄

## GTZAN Dataset Genres
1. Blues
2. Classical
3. Country
4. Disco
5. Hip-hop
6. Jazz
7. Metal
8. Pop
9. Reggae
10. Rock