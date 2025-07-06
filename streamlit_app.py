import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import predict_genre, validate_audio_file, get_model_info, download_sample_data
import tempfile

# Page configuration
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide"
)

def main():
    st.title("🎵 Music Genre Classifier")
    st.markdown("Upload an audio file to predict its music genre using deep learning!")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.markdown("""
        This app uses a Convolutional Neural Network (CNN) trained on the GTZAN dataset 
        to classify music into 10 different genres:
        
        - Blues
        - Classical  
        - Country
        - Disco
        - Hip-hop
        - Jazz
        - Metal
        - Pop
        - Reggae
        - Rock
        """)
        
        st.header("Model Info")
        show_model_info()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🎯 Predict", "📊 Dataset Info", "🔧 Training"])
    
    with tab1:
        prediction_tab()
    
    with tab2:
        dataset_info_tab()
        
    with tab3:
        training_tab()

def prediction_tab():
    st.header("Genre Prediction")
    
    # Check if model exists
    model_path = 'models/genre_classifier.h5'
    encoder_path = 'models/label_encoder.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(encoder_path):
        st.error("❌ Model not found! Please train the model first.")
        st.info("Go to the 'Training' tab to train the model on the GTZAN dataset.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'flac', 'm4a'],
        help="Upload a music file to classify its genre"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.write(f"**Filename:** {uploaded_file.name}")
        st.write(f"**File size:** {uploaded_file.size} bytes")
        
        # Play audio
        st.audio(uploaded_file)
        
        # Save temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name
        
        # Predict button
        if st.button("🎯 Predict Genre", type="primary"):
            try:
                with st.spinner("Analyzing audio... This may take a few seconds."):
                    predicted_genre, confidence, top_predictions = predict_genre(
                        tmp_file_path, model_path, encoder_path
                    )
                
                if predicted_genre is None:
                    st.error("❌ Could not process the audio file. Please try another file.")
                else:
                    # Display results
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.success(f"**Predicted Genre:** {predicted_genre.upper()}")
                        st.info(f"**Confidence:** {confidence:.1%}")
                    
                    with col2:
                        # Create visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        genres = [pred[0] for pred in top_predictions]
                        confidences = [pred[1] for pred in top_predictions]
                        
                        bars = ax.barh(genres, confidences, color=sns.color_palette("viridis", len(genres)))
                        ax.set_xlabel('Confidence')
                        ax.set_title('Top 3 Genre Predictions')
                        ax.set_xlim(0, 1)
                        
                        # Add percentage labels
                        for i, (bar, conf) in enumerate(zip(bars, confidences)):
                            ax.text(conf + 0.01, i, f'{conf:.1%}', 
                                   va='center', fontweight='bold')
                        
                        st.pyplot(fig)
                        plt.close()
                    
                    # Detailed results
                    st.subheader("📊 Detailed Results")
                    results_df = st.dataframe({
                        'Rank': range(1, len(top_predictions) + 1),
                        'Genre': [pred[0].title() for pred in top_predictions],
                        'Confidence': [f"{pred[1]:.2%}" for pred in top_predictions]
                    }, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

def dataset_info_tab():
    st.header("📊 GTZAN Dataset Information")
    
    st.markdown("""
    The GTZAN dataset is a collection of 1000 audio tracks, each 30 seconds long. 
    It contains 10 genres, each represented by 100 tracks.
    """)
    
    # Dataset download instructions
    st.subheader("📥 Download Instructions")
    instructions = download_sample_data()
    st.code(instructions, language="text")
    
    # Show data structure if available
    data_path = 'data/genres_original'
    if os.path.exists(data_path):
        st.subheader("✅ Dataset Status")
        st.success("Dataset found!")
        
        # Count files in each genre
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                 'jazz', 'metal', 'pop', 'reggae', 'rock']
        
        genre_counts = {}
        for genre in genres:
            genre_path = os.path.join(data_path, genre)
            if os.path.exists(genre_path):
                count = len([f for f in os.listdir(genre_path) if f.endswith('.wav')])
                genre_counts[genre] = count
        
        if genre_counts:
            st.subheader("📈 Files per Genre")
            st.bar_chart(genre_counts)
            
            total_files = sum(genre_counts.values())
            st.metric("Total Audio Files", total_files)
    else:
        st.warning("⚠️ Dataset not found. Please download and extract the GTZAN dataset.")

def training_tab():
    st.header("🔧 Model Training")
    
    data_path = 'data/genres_original'
    model_path = 'models/genre_classifier.h5'
    
    if not os.path.exists(data_path):
        st.warning("⚠️ Dataset not found. Please download the GTZAN dataset first.")
        st.markdown("See the 'Dataset Info' tab for download instructions.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Status")
        st.success("✅ Dataset found!")
        
        # Count total files
        total_files = 0
        for genre in os.listdir(data_path):
            genre_path = os.path.join(data_path, genre)
            if os.path.isdir(genre_path):
                total_files += len([f for f in os.listdir(genre_path) if f.endswith('.wav')])
        
        st.metric("Total Audio Files", total_files)
    
    with col2:
        st.subheader("Model Status")
        if os.path.exists(model_path):
            st.success("✅ Trained model found!")
            
            # Show test results if available
            test_results_path = 'models/test_results.txt'
            if os.path.exists(test_results_path):
                with open(test_results_path, 'r') as f:
                    results = f.read()
                st.text(results)
        else:
            st.warning("⚠️ No trained model found")
    
    # Training button
    st.subheader("Start Training")
    st.markdown("""
    **Note:** Training can take several hours depending on your hardware. 
    The model will be saved to `models/genre_classifier.h5` when complete.
    """)
    
    if st.button("🚀 Start Training", type="primary"):
        if not os.path.exists(data_path):
            st.error("Cannot start training: Dataset not found!")
            return
        
        st.warning("⚠️ Training should be run from the command line for better performance:")
        st.code("python cli_app.py train", language="bash")
        st.markdown("This ensures proper logging and prevents timeout issues.")

def show_model_info():
    model_path = 'models/genre_classifier.h5'
    
    if os.path.exists(model_path):
        try:
            info = get_model_info(model_path)
            if info:
                st.success("✅ Model loaded")
                st.text(f"Parameters: {info['total_params']:,}")
                st.text(f"Layers: {info['layers']}")
        except:
            st.error("❌ Model error")
    else:
        st.warning("⚠️ No model found")

if __name__ == "__main__":
    main()