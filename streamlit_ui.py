import streamlit as st
import os
import glob
import base64
import numpy as np
import re
from ensemble_predict import EnsemblePredictor

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
    }
    .card-back ul {
        list-style-position: inside;
        width: 100%;
        padding: 0 20px; /* Add some padding on the sides */
    }
    .card-back li {
        display: flex;
        justify-content: space-between;
        width: 100%;
        margin-bottom: 10px;
        font-size: 18px;
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
    return EnsemblePredictor()

predictor = load_predictor()

# Load test data
@st.cache_data
def load_test_data():
    with np.load('processed_solar_data.npz') as data:
        return data['X_test']

X_test = load_test_data()

# Find saliency maps
saliency_dir = "saliency_exports"
saliency_files = sorted(glob.glob(os.path.join(saliency_dir, "*.png")))

# Initialize session state for card flip status
if 'flipped' not in st.session_state:
    st.session_state.flipped = {f"card_{i}": False for i in range(15)}
if 'saliency_images' not in st.session_state:
    st.session_state.saliency_images = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}

# Create a container for the cards
st.markdown('<div class="card-container">', unsafe_allow_html=True)

# Create cards in rows of 5 for better layout control
cards_per_row = 5
total_cards = 15

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

            # Get saliency image for the card if not already set
            if card_key not in st.session_state.saliency_images:
                if saliency_files:
                    # Pick an image that hasn't been used yet from the available files
                    used_images = set(st.session_state.saliency_images.values())
                    available_images = [f for f in saliency_files if f not in used_images]
                    
                    if available_images:
                        image_path = available_images[0]
                    else:
                        # If all unique images are used, cycle through them
                        image_path = saliency_files[card_idx % len(saliency_files)]
                    st.session_state.saliency_images[card_key] = image_path
                    
                    # Get sample index from filename and make prediction
                    match = re.search(r'sample_(\d+)', image_path)
                    if match:
                        sample_idx = int(match.group(1))
                        if sample_idx < len(X_test):
                            X_sample = X_test[sample_idx]
                            ensemble_pred, individual_preds = predictor.predict_single_sample(X_sample)
                            
                            # Add ensemble prediction to the dict
                            individual_preds['Ensemble'] = ensemble_pred
                            st.session_state.predictions[card_key] = individual_preds
                else:
                    st.session_state.saliency_images[card_key] = None
            
            image_path = st.session_state.saliency_images.get(card_key)
            image_b64 = get_image_base64(image_path)
            predictions = st.session_state.predictions.get(card_key)

            # Render card with current flip state
            flip_class = "card-inner flipped" if st.session_state.flipped[card_key] else "card-inner"
            
            card_front_content = f'<img src="data:image/png;base64,{image_b64}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 8px;">' if image_b64 else f'<p>Card {card_idx+1}</p><p style="font-size: 14px;">No Saliency Map</p>'

            if predictions:
                # Define the desired order
                model_order = ['Bp', 'Br', 'Bt', 'continuum', 'Ensemble']
                # Create list items, ensuring the order
                prediction_items = "".join([f"<li><span>{model_name}:</span> <span>{predictions.get(model_name, 0):.2f}</span></li>" for model_name in model_order if model_name in predictions])
                card_back_content = f"<ul>{prediction_items}</ul>"
            else:
                card_back_content = "<p>No predictions available.</p>"

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
            
            # Check button state and update flip state if clicked
            if st.button("See Predictions", key=f"toggle_{card_idx}"):
                st.session_state.flipped[card_key] = not st.session_state.flipped[card_key]
                st.rerun()  # Force rerun to update the display immediately
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Add minimal spacing between rows
    if row < total_cards - cards_per_row:  # Don't add spacing after last row
        st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)