import streamlit as st
import os
import glob
import base64
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

sys.path.append('Running Models')
sys.path.append(os.path.join(os.getcwd(), 'Visualizing Data'))
from ensemble_predict import EnsemblePredictor, load_test_data

# Simple saliency map generation function
def create_saliency_map(model, img_array):
    """Create a saliency map for a given model and image"""
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32) if isinstance(img_array, np.ndarray) else img_array
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        loss = model(img_tensor)[0][0]
    grads = tf.math.abs(tape.gradient(loss, img_tensor))
    if len(grads.shape) == 5: 
        grads = tf.reduce_mean(grads, axis=1)
    grads = tf.reduce_max(grads, axis=-1) if grads.shape[-1] > 1 else tf.squeeze(grads, axis=-1)
    return (grads / (tf.reduce_max(grads) + 1e-8))[0].numpy()

def generate_saliency_image(model, X_sample, channel_name, sample_idx):
    """Generate a saliency map image and return as base64 string"""
    try:
        channel_data_batch = np.expand_dims(X_sample, axis=0)
        channel_data_tensor = tf.convert_to_tensor(channel_data_batch, dtype=tf.float32)
        
        # Create saliency map
        saliency = create_saliency_map(model, channel_data_tensor)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(saliency, cmap='hot')
        ax.set_title(f'{channel_name} - Sample {sample_idx}')
        ax.axis('off')
        
        # Convert to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close(fig)
        
        return img_base64
    except Exception as e:
        print(f"Error generating saliency for {channel_name}: {e}")
        return None

st.set_page_config(layout="wide")

def get_image_base64(path):
    if path and os.path.exists(path):
        with open(path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

st.markdown("""
    <style>
    .card-container {
        padding: 20px;
    }
    .card-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 30px;
        padding: 10px;
    }
    .card {
        width: 280px; /* 2x scale: 140px * 2 */
        height: 400px; /* 2x scale: 200px * 2 */
        perspective: 1000px;
        margin: 0 auto 16px auto; /* 2x scale: 8px * 2 */
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .card {
            width: 85vw; /* Take up most of screen width on mobile */
            height: calc(85vw * 1.43); /* Maintain aspect ratio (400/280 = 1.43) */
            max-width: 320px; /* Maximum size limit */
            max-height: 460px;
        }
        .card-container {
            padding: 10px;
        }
        .card-wrapper {
            margin-bottom: 20px;
        }
    }
    .card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        transition: transform 0.6s;
        transform-style: preserve-3d;
    }
    .card-inner.flipped {
        transform: rotateY(180deg);
    }
    .card-front, .card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        backface-visibility: hidden;
        border-radius: 8px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .card-front {
        background-color: #d3d3d3;
        color: #333;
        font-size: 28px; /* 2x scale: 14px * 2 */
        padding: 0;
        overflow: hidden;
    }
    .card-back {
        background-color: #333;
        color: #fff;
        transform: rotateY(180deg);
        padding: 16px; /* 2x scale: 8px * 2 */
        font-size: 20px; /* 2x scale: 10px * 2 */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    
    /* Mobile font scaling */
    @media (max-width: 768px) {
        .card-front {
            font-size: 24px;
        }
        .card-back {
            font-size: 16px;
            padding: 12px;
        }
    }
    .card-back p, .card-back ul {
        margin: 0;
        padding: 0;
        text-align: center;
        width: 100%;
    }
    .card-back ul {
        list-style: none;
        width: 100%;
        max-width: 220px;
        padding: 0;
        margin: 0 auto;
        text-align: left;
    }
    .card-back li {
        display: flex;
        justify-content: space-between;
        align-items: center;
        width: 100%;
        margin-bottom: 8px;
        font-size: 16px;
        font-weight: bold;
        padding: 0 10px;
    }
    .card-back li span:first-child {
        color: #ccc;
        font-size: 14px;
    }
    .card-back li span:last-child {
        color: #fff;
        font-weight: bold;
        font-size: 14px;
    }
    /* Custom styling for button centering */
    .stButton > button {
        width: 200px; /* 2x scale: 100px * 2 */
        margin: 0 auto;
        display: block;
        font-size: 22px; /* 2x scale: 11px * 2 */
        padding: 8px 16px; /* 2x scale: 4px 8px * 2 */
    }
    .stButton {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    /* Reduce spacing between columns */
    [data-testid="column"] {
        padding: 0 10px; /* 2x scale: 5px * 2 */
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .element-container {
        margin-bottom: 10px; /* 2x scale: 5px * 2 */
        width: 100%;
        display: flex;
        justify-content: center;
    }
    
    .card-column-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }

    /* Mobile button adjustments */
    @media (max-width: 1500px) {
        .stButton > button {
            width: 180px;
            font-size: 18px;
            padding: 6px 12px;
        }
        [data-testid="column"] {
            padding: 0 5px;
            width: auto !important; /* Let column width be determined by content */
            flex: none !important;
            margin-bottom: 1rem;
        }
        .element-container {
            margin-bottom: 8px;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
            align-items: center; /* Center the columns */
        }
        /* Ensure cards stack vertically on mobile */
        .card-container {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
    }
    .debug {
        color: green;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Model Prediction Cards")

# Load the ensemble predictor
@st.cache_resource
def load_predictor():
    try:
        return EnsemblePredictor()
    except Exception as e:
        st.error(f"Failed to load predictor: {e}")
        return None

predictor = load_predictor()

# Check if predictor loaded successfully
if predictor is None:
    st.error("Failed to load the ensemble predictor. Please check your model files.")
    st.stop()

# Load test data using the ensemble_predict function
@st.cache_data
def load_test_data_ui():
    try:
        X_test, y_test = load_test_data()
        if X_test is not None and y_test is not None:
            # Sample up to 15 samples (or all if fewer than 15)
            total_samples = len(X_test)
            sample_size = min(15, total_samples)
            if total_samples >= sample_size:
                random_indices = np.random.choice(total_samples, size=sample_size, replace=False)
                return X_test[random_indices], y_test[random_indices]
            else:
                return X_test, y_test
        else:
            # Fallback to hardcoded path if ensemble function fails
            fallback_path = 'processed_solar_data.npz'
            if os.path.exists(fallback_path):
                with np.load(fallback_path) as data:
                    X_test = data['X_test']
                    y_test = data.get('y_test', None)
                    # Sample up to 15 samples (or all if fewer than 15)
                    total_samples = len(X_test)
                    sample_size = min(15, total_samples)
                    if total_samples >= sample_size:
                        random_indices = np.random.choice(total_samples, size=sample_size, replace=False)
                        X_test = X_test[random_indices]
                        if y_test is not None:
                            y_test = y_test[random_indices]
                    return X_test, y_test
            else:
                st.error("No test data found. Please check your data paths.")
                return None, None
    except Exception as e:
        st.error(f"Failed to load test data: {e}")
        return None, None

test_data = load_test_data_ui()
if test_data[0] is not None:
    X_test, y_test = test_data
else:
    st.error("Cannot proceed without test data")
    st.stop()


def get_predictions(sample_idx, predictor, X_test):
    """Safely get predictions for a sample, with error handling"""
    try:
        if sample_idx >= len(X_test):
            return {'Error': f'Sample index {sample_idx} out of range'}
        
        X_sample = X_test[sample_idx]
        ensemble_pred, individual_preds = predictor.predict_single_sample(X_sample)
        
        # Add ensemble prediction to the dict
        individual_preds['Ensemble'] = ensemble_pred
        return individual_preds
    except Exception as e:
        return {'Error': f'Prediction failed: {str(e)}'}

@st.cache_data
def generate_all_saliency_maps(X_test, _predictor):
    """Generate saliency maps for all samples using the first available model"""
    saliency_maps = {}
    if _predictor and hasattr(_predictor, 'models') and _predictor.models:
        # Use the first available model for saliency generation
        first_model_name = list(_predictor.models.keys())[0]
        first_model = _predictor.models[first_model_name]
        channel_idx = _predictor.channel_names.index(first_model_name)
        
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        for i in range(len(X_test)):
            progress_text.text(f"Generating saliency map {i+1}/{len(X_test)}...")
            progress_bar.progress((i + 1) / len(X_test))
            
            # Extract channel data for this sample
            channel_data = X_test[i][:, :, :, channel_idx:channel_idx+1]
            saliency_base64 = generate_saliency_image(first_model, channel_data, first_model_name, i)
            saliency_maps[i] = saliency_base64
        
        progress_bar.empty()
        progress_text.empty()
    
    return saliency_maps

# Generate saliency maps for all samples
actual_sample_count = len(X_test)
st.info(f"Generating saliency maps for all {actual_sample_count} samples...")
with st.spinner("This may take a moment..."):
    saliency_maps = generate_all_saliency_maps(X_test, predictor)
st.success("Saliency maps generated!")

# Initialize session state for card flip status and predictions
if 'flipped' not in st.session_state:
    st.session_state.flipped = {f"card_{i}": False for i in range(actual_sample_count)}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Create a container for the cards
st.markdown('<div class="card-container">', unsafe_allow_html=True)

# Create cards in rows of 5 for better layout control
cards_per_row = 5
total_cards = actual_sample_count

# Add responsive behavior with JavaScript
st.markdown("""
<script>
function checkMobile() {
    return window.innerWidth <= 768;
}
</script>
""", unsafe_allow_html=True)

for row in range(0, total_cards, cards_per_row):
    # Create columns for this row with equal spacing
    # On mobile, we'll use single column layout via CSS
    cols = st.columns(cards_per_row, gap="small")
    
    for col_idx in range(cards_per_row):
        card_idx = row + col_idx
        if card_idx >= total_cards:
            break
            
        card_key = f"card_{card_idx}"
        
        with cols[col_idx]:
            # Create a container to center everything with responsive class
            st.markdown(f'<div class="card-column-wrapper">', unsafe_allow_html=True)

            # Get saliency map for this card
            saliency_base64 = saliency_maps.get(card_idx)
            
            # Handle prediction processing - simplified logic
            predictions = st.session_state.predictions.get(card_idx)
            has_predictions = predictions is not None

            # Render card with current flip state - only flip if predictions are complete
            predictions_complete = (predictions is not None and 'Error' not in predictions)
            
            # Only allow flipping if predictions are complete
            should_flip = st.session_state.flipped[card_key] and predictions_complete
            flip_class = "card-inner flipped" if should_flip else "card-inner"
            
            if saliency_base64:
                card_front_content = f'<img src="data:image/png;base64,{saliency_base64}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">'
            else:
                card_front_content = f'<p>Card {card_idx+1}</p><p style="font-size: 14px;">No Saliency Map</p>'

            if predictions:
                # Check if there's an error
                if 'Error' in predictions:
                    card_back_content = f"<p style='color: #ff6b6b; text-align: center;'>{predictions['Error']}</p>"
                else:
                    # Define the desired order
                    model_order = ['Bp', 'Br', 'Bt', 'continuum', 'Ensemble']
                    # Create list items with FLARE/NO FLARE classification
                    prediction_items = "".join([
                        f"<li><span>{model_name}:</span> <span>{'FLARE' if predictions.get(model_name, 0) > 0.5 else 'NO FLARE'}</span></li>" 
                        for model_name in model_order if model_name in predictions
                    ])
                    
                    # Add actual value if available
                    actual_value = ""
                    if card_idx < len(y_test) and y_test is not None:
                        actual_class = 'FLARE' if y_test[card_idx] == 1 else 'NO FLARE'
                        actual_value = f"<li style='border-top: 1px solid #666; margin-top: 8px; padding-top: 8px;'><span>Actual:</span> <span>{actual_class}</span></li>"
                    
                    card_back_content = f"<ul>{prediction_items}{actual_value}</ul>"
            else:
                card_back_content = "<p style='text-align: center;'>Click 'See Predictions' to load predictions.</p>"

            st.markdown(f"""
                <div class="card-wrapper">
                    <div class="card">
                        <div class="{flip_class}">
                            <div class="card-front">
                                {card_front_content}
                            </div>
                            <div class="card-back">
                                {card_back_content}
                            </div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Button logic - process predictions on demand
            has_predictions = card_idx in st.session_state.predictions
            button_text = "See Predictions"
            
            if st.button(button_text, key=f"toggle_{card_idx}"):
                if not has_predictions:
                    # Process prediction immediately
                    with st.spinner(f"Processing sample {card_idx + 1}..."):
                        prediction_result = get_predictions(card_idx, predictor, X_test)
                        st.session_state.predictions[card_idx] = prediction_result
                    # Flip the card to show results
                    st.session_state.flipped[card_key] = True
                else:
                    # Toggle flip state if predictions already exist
                    st.session_state.flipped[card_key] = not st.session_state.flipped[card_key]
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add minimal spacing between rows
    if row < total_cards - cards_per_row:  # Don't add spacing after last row
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)