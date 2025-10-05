import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="NASA Exoplanet Hunter",
    page_icon="🚀",
    layout="wide"
)

# Updated CSS for white text
st.markdown("""
<style>
    :root {
        --nasa-blue: #0B3D91;
        --nasa-red: #FC3D21;
        --nasa-white: #FFFFFF;
        --nasa-dark-blue: #062561;
        --nasa-light-gray: #1E1E1E;
        --nasa-dark-gray: #CCCCCC;
    }
    
    /* Make ALL text white */
    .main-header, .sub-header, .stMarkdown, h1, h2, h3, h4, h5, h6,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    .streamlit-expanderHeader, .st-c7, .st-bw, .st-bv, .st-bu,
    .stAlert, .st-bb, .st-ba, .st-b9, .st-b8, .st-b7, .st-b6,
    p, div, span, label {
        color: white !important;
    }
    
    /* Dark background */
    .main {
        background-color: #0a0a0a;
    }
    
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1a1a1a;
    }
    
    /* Rest of your NASA styling... */
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🚀 NASA Exoplanet Hunter</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Exoplanet Classification using NASA\'s Kepler, K2, and TESS Data</p>', unsafe_allow_html=True)

def derive_features(input_data):
    """Derive 21 additional features from the 10 base features"""
    derived = input_data.copy()
    
    def safe_divide(a, b):
        return np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    
    derived["radius_ratio"] = safe_divide(derived["planet_radius"], derived["star_radius"])
    derived["a_over_rstar"] = safe_divide(derived["semi_major_axis"], derived["star_radius"])
    derived["transit_signal"] = derived["transit_depth"] * np.sqrt(np.abs(derived["transit_duration"]))
    derived["star_density_proxy"] = safe_divide(derived["star_gravity"], derived["star_radius"])

    for col in ["orbital_period", "transit_depth", "planet_radius"]:
        derived[f"log_{col}"] = np.log1p(np.abs(derived[col]))

    derived["duty_cycle"] = safe_divide(derived["transit_duration"], derived["orbital_period"])
    derived["normalized_depth"] = safe_divide(derived["transit_depth"], (derived["star_radius"]**2))
    derived["luminosity_proxy"] = (derived["star_radius"]**2) * (derived["star_temp"]**4)
    derived["depth_to_duration"] = safe_divide(derived["transit_depth"], derived["transit_duration"])
    derived["snr_per_duration"] = safe_divide(derived["snr"], derived["transit_duration"])
    derived["log_luminosity"] = np.log1p(np.abs(derived["luminosity_proxy"]))
    derived["scaled_a_rstar"] = safe_divide(derived["semi_major_axis"], derived["star_radius"])
    derived["normalized_snr"] = safe_divide(derived["snr"], np.sqrt(np.abs(derived["orbital_period"])))
    derived["luminosity_star"] = (derived["star_radius"] ** 2) * (derived["star_temp"] / 5778) ** 4
    derived["eq_temp"] = derived["star_temp"] * np.sqrt(safe_divide(derived["star_radius"], (2 * derived["semi_major_axis"]))) * ((1 - 0.3) ** 0.25)
    derived["insolation_flux"] = safe_divide(derived["luminosity_star"], (derived["semi_major_axis"] ** 2))
    derived["snr_scaled_lum"] = safe_divide(derived["snr"], np.sqrt(np.abs(derived["luminosity_star"])))
    derived["depth_temp_ratio"] = safe_divide(derived["transit_depth"], derived["star_temp"])
    derived["radius_eqtemp_ratio"] = safe_divide(derived["planet_radius"], np.sqrt(np.abs(derived["eq_temp"])))
    
    derived = derived.replace([np.inf, -np.inf], 0)
    
    return derived

@st.cache_resource
def load_artifacts():
    """Load model and artifacts"""
    try:
        model = joblib.load('xgb_model.pkl')
        encoder = joblib.load('xgb_label_encoder.pkl')
        
        if hasattr(model, 'feature_names_in_'):
            all_features = list(model.feature_names_in_)
        else:
            all_features = [
                'orbital_period', 'transit_duration', 'transit_depth', 'planet_radius',
                'semi_major_axis', 'impact', 'star_temp', 'star_radius', 'star_gravity', 'snr',
                'radius_ratio', 'a_over_rstar', 'transit_signal', 'star_density_proxy',
                'log_orbital_period', 'log_transit_depth', 'log_planet_radius', 'duty_cycle',
                'normalized_depth', 'luminosity_proxy', 'depth_to_duration', 'snr_per_duration',
                'log_luminosity', 'scaled_a_rstar', 'normalized_snr', 'luminosity_star',
                'eq_temp', 'insolation_flux', 'snr_scaled_lum', 'depth_temp_ratio', 'radius_eqtemp_ratio'
            ]
            
        return model, encoder, all_features
    except FileNotFoundError as e:
        st.error(f"❌ Model artifacts not found: {e}")
        return None, None, None

def main():
    # Load artifacts
    model, encoder, all_features = load_artifacts()
    
    if model is None:
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📊 Model Performance")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "75.7%")
            st.metric("F1-Score", "75.5%")
        with col2:
            st.metric("Balanced Acc", "75.0%")
            st.metric("Precision", "76.3%")
        
        st.progress(0.757)
        st.caption("Overall performance: 75.7%")
        
        with st.expander("📈 Class-wise Performance"):
            st.metric("CANDIDATE F1", "75.6%")
            st.metric("CONFIRMED F1", "73.5%") 
            st.metric("FALSE POSITIVE F1", "77.3%")
        
        st.markdown("---")
        st.markdown("### 🎯 Classification")
        for class_name in encoder.classes_:
            st.write(f"• {class_name}")
        
        st.markdown("### 🔧 Features")
        st.write(f"**10 base + 21 derived**")

    # Main content tabs
    tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction"])
    
    with tab1:
        st.header("Single Exoplanet Prediction")
        st.write("Enter the 10 base features - the app will automatically derive 21 additional features for prediction")
        
        input_data = {}
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Orbital Features")
            input_data['orbital_period'] = st.number_input(
                "Orbital Period (days)", 
                min_value=0.1, value=10.0, step=0.1
            )
            input_data['transit_duration'] = st.number_input(
                "Transit Duration (hours)", 
                min_value=0.1, value=5.0, step=0.1
            )
            input_data['transit_depth'] = st.number_input(
                "Transit Depth (ppm)", 
                min_value=0, value=5000, step=100
            )
            input_data['planet_radius'] = st.number_input(
                "Planet Radius (Earth radii)", 
                min_value=0.1, value=1.5, step=0.1
            )
        
        with col2:
            st.subheader("🌌 System Geometry")
            input_data['semi_major_axis'] = st.number_input(
                "Semi-major Axis (AU)", 
                min_value=0.01, value=1.0, step=0.01
            )
            input_data['impact'] = st.number_input(
                "Impact Parameter", 
                min_value=0.0, max_value=1.0, value=0.5, step=0.01
            )
            input_data['snr'] = st.number_input(
                "Signal-to-Noise Ratio", 
                min_value=0.0, value=15.0, step=0.1
            )
        
        with col3:
            st.subheader("⭐ Stellar Properties")
            input_data['star_temp'] = st.number_input(
                "Star Temperature (K)", 
                min_value=2000, value=5778, step=50
            )
            input_data['star_radius'] = st.number_input(
                "Star Radius (Solar radii)", 
                min_value=0.1, value=1.0, step=0.1
            )
            input_data['star_gravity'] = st.number_input(
                "Star Gravity (log g)", 
                min_value=3.0, max_value=5.0, value=4.4, step=0.1
            )
        
        if st.button("🚀 Classify Exoplanet", type="primary", use_container_width=True):
            try:
                base_df = pd.DataFrame([input_data])
                
                with st.spinner("🔄 Deriving features and making prediction..."):
                    full_df = derive_features(base_df)
                    final_features = full_df[all_features]
                    
                    prediction_encoded = model.predict(final_features)[0]
                    prediction_proba = model.predict_proba(final_features)[0]
                    prediction = encoder.inverse_transform([prediction_encoded])[0]
                
                st.success(f"**Prediction: {prediction}**")
                
                st.subheader("Classification Confidence")
                fig, ax = plt.subplots(figsize=(10, 4))
                classes = encoder.classes_
                y_pos = np.arange(len(classes))
                
                bars = ax.barh(y_pos, prediction_proba, color=['#0B3D91', '#FC3D21', '#062561'])
                ax.set_yticks(y_pos)
                ax.set_yticklabels(classes)
                ax.set_xlabel('Probability')
                ax.set_title('Classification Probabilities')
                ax.set_xlim(0, 1)
                
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{width:.2%}', ha='left', va='center')
                
                st.pyplot(fig)
                
                if prediction == "CONFIRMED":
                    st.info("🎉 **Confirmed Exoplanet**: High confidence (73.5% F1-score) - This object has been validated as a real exoplanet!")
                elif prediction == "CANDIDATE":
                    st.info("🔍 **Planetary Candidate**: Good confidence (75.6% F1-score) - This shows strong signs of being an exoplanet but requires further validation.")
                else:
                    st.info("❌ **False Positive**: Highest confidence (77.3% F1-score) - This object is likely not an exoplanet")
                    
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    with tab2:
        st.header("Batch Prediction from CSV")
        st.info("Upload a CSV with the 10 base features - the app will derive 21 additional features automatically")
        
        base_feature_names = [
            'orbital_period', 'transit_duration', 'transit_depth', 'planet_radius',
            'semi_major_axis', 'impact', 'star_temp', 'star_radius', 'star_gravity', 'snr'
        ]
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                missing_features = set(base_feature_names) - set(df.columns)
                if missing_features:
                    st.error(f"❌ Missing required base features: {list(missing_features)}")
                else:
                    if st.button("Run Batch Prediction", type="primary"):
                        with st.spinner("Processing batch data..."):
                            full_data = derive_features(df)
                            final_features = full_data[all_features]
                            
                            predictions = model.predict(final_features)
                            probabilities = model.predict_proba(final_features)
                            
                            result_df = df.copy()
                            result_df['prediction'] = encoder.inverse_transform(predictions)
                            
                            for i, class_name in enumerate(encoder.classes_):
                                result_df[f'probability_{class_name}'] = probabilities[:, i]
                            
                            st.success(f"✅ Successfully processed {len(result_df)} candidates!")
                            
                            st.subheader("Prediction Summary")
                            summary = result_df['prediction'].value_counts()
                            st.bar_chart(summary)
                            
                            st.subheader("Detailed Results")
                            st.dataframe(result_df)
                            
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                "📥 Download Predictions as CSV",
                                csv,
                                "exoplanet_predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()