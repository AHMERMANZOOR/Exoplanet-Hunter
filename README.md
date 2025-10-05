# 🚀 NASA Exoplanet Hunter

**AI-Powered Exoplanet Classification using NASA's Kepler, K2, and TESS Mission Data**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app)
[![NASA Open Source](https://img.shields.io/badge/NASA-Open%20Source-blue.svg)](https://www.nasa.gov/open-source)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌌 About

An intelligent machine learning system that classifies celestial objects as **Confirmed Exoplanets**, **Planetary Candidates**, or **False Positives** using NASA's open-source data. Built for the NASA Space Apps Challenge 2024, this tool brings exoplanet discovery capabilities to researchers and astronomy enthusiasts worldwide.

## 🎯 Features

- **AI-Powered Classification**: 75.7% accurate XGBoost model trained on real NASA data
- **Automated Feature Engineering**: Converts 10 basic astronomical measurements into 31 scientific features
- **User-Friendly Interface**: Clean Streamlit web app for both single and batch predictions
- **Real NASA Data**: Trained on combined datasets from Kepler, K2, and TESS missions
- **Professional Results**: Confidence scores and detailed classification reports

## 🔬 Model Performance

| Metric | Score |
|--------|-------|
| Validation Accuracy | 75.7% |
| Balanced Accuracy | 75.0% |
| F1-Score (Macro) | 75.5% |
| **Best Class** (False Positive) | **77.3%** |

## 🛠 Quick Start

1. **Access the Web App**: [Live Demo](https://your-app-link.streamlit.app)
2. **Single Prediction**: Enter 10 basic astronomical parameters
3. **Batch Analysis**: Upload CSV files for multiple candidates
4. **Get Results**: Instant classifications with confidence scores

## 📊 Input Features

**10 Base Parameters Required:**
- Orbital Period (days)
- Transit Duration (hours) 
- Transit Depth (ppm)
- Planet Radius (Earth radii)
- Semi-major Axis (AU)
- Impact Parameter
- Star Temperature (K)
- Star Radius (Solar radii)
- Star Gravity (log g)
- Signal-to-Noise Ratio

## 🏗 Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/nasa-exoplanet-hunter.git

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py

