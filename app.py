"""
Yield Monitoring using UAV - Streamlit Application
Academic Prototype for Crop Health and Yield Estimation using RGB Imagery

This application implements advanced vegetation indices suitable for RGB images
without NIR band, using adaptive thresholding and spatial analysis techniques.

Author: Academic Project Prototype
Date: February 2026
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
import io

# ============================================================================
# CONFIGURATION AND CONSTANTS
# ============================================================================

# Maximum image dimensions for processing (to prevent browser crashes)
MAX_IMAGE_WIDTH = 1920
MAX_IMAGE_HEIGHT = 1080

# Health classification thresholds (percentile-based, adaptive)
HEALTH_PERCENTILES = {
    'poor': 33,      # Bottom 33rd percentile
    'moderate': 67   # 33rd to 67th percentile
    # 'healthy': Above 67th percentile
}

# Yield estimation parameters (kg per sq meter based on health)
YIELD_COEFFICIENTS = {
    'healthy': 0.65,    # High productivity
    'moderate': 0.40,   # Medium productivity
    'poor': 0.15        # Low productivity
}

# Color mapping for visualization (BGR format for OpenCV)
HEALTH_COLORS = {
    'healthy': (0, 255, 0),      # Green
    'moderate': (0, 165, 255),   # Orange
    'poor': (0, 0, 255)          # Red
}

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def resize_image_if_needed(image, max_width=MAX_IMAGE_WIDTH, max_height=MAX_IMAGE_HEIGHT):
    """
    Resize image if it exceeds maximum dimensions to prevent browser memory issues.
    
    Args:
        image: Input image (numpy array)
        max_width: Maximum allowed width
        max_height: Maximum allowed height
    
    Returns:
        Resized image and scale factor
    """
    height, width = image.shape[:2]
    
    # Check if resizing is needed
    if width <= max_width and height <= max_height:
        return image, 1.0
    
    # Calculate scale factor
    scale_w = max_width / width
    scale_h = max_height / height
    scale = min(scale_w, scale_h)
    
    # Calculate new dimensions
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return resized, scale


# ============================================================================
# VEGETATION INDEX CALCULATIONS
# ============================================================================

def calculate_exg(image):
    """
    Calculate Excess Green Index (ExG)
    
    ExG = 2*G - R - B
    
    This index emphasizes green vegetation by comparing the green channel
    against red and blue. It's particularly effective for early-stage crops
    and provides good contrast between vegetation and soil.
    
    Args:
        image: RGB image (numpy array, uint8)
    
    Returns:
        ExG index normalized to [0, 1]
    """
    # Convert to float for calculation
    img_float = image.astype(np.float32) / 255.0
    
    # Extract channels
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    
    # Calculate ExG
    exg = 2 * G - R - B
    
    # Normalize to [0, 1] range
    exg_normalized = (exg - exg.min()) / (exg.max() - exg.min() + 1e-8)
    
    return exg_normalized


def calculate_vari(image):
    """
    Calculate Visible Atmospherically Resistant Index (VARI)
    
    VARI = (G - R) / (G + R - B)
    
    VARI is designed to be relatively insensitive to atmospheric effects
    and provides robust vegetation detection across varying lighting conditions.
    It's particularly useful for UAV imagery taken at different times of day.
    
    Args:
        image: RGB image (numpy array, uint8)
    
    Returns:
        VARI index normalized to [0, 1]
    """
    # Convert to float for calculation
    img_float = image.astype(np.float32) / 255.0
    
    # Extract channels
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    
    # Calculate VARI with epsilon to avoid division by zero
    denominator = G + R - B + 1e-8
    vari = (G - R) / denominator
    
    # Normalize to [0, 1] range
    vari_normalized = (vari - vari.min()) / (vari.max() - vari.min() + 1e-8)
    
    return vari_normalized


def calculate_gli(image):
    """
    Calculate Green Leaf Index (GLI)
    
    GLI = (2*G - R - B) / (2*G + R + B)
    
    GLI is a normalized version of ExG, providing better contrast
    and reducing sensitivity to illumination variations.
    
    Args:
        image: RGB image (numpy array, uint8)
    
    Returns:
        GLI index normalized to [0, 1]
    """
    # Convert to float for calculation
    img_float = image.astype(np.float32) / 255.0
    
    # Extract channels
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    B = img_float[:, :, 2]
    
    # Calculate GLI with epsilon to avoid division by zero
    numerator = 2 * G - R - B
    denominator = 2 * G + R + B + 1e-8
    gli = numerator / denominator
    
    # Normalize to [0, 1] range
    gli_normalized = (gli - gli.min()) / (gli.max() - gli.min() + 1e-8)
    
    return gli_normalized


def calculate_rgb_ndvi(image):
    """
    Calculate RGB-based pseudo-NDVI
    
    RGB-NDVI = (G - R) / (G + R)
    
    This is an approximation of NDVI using visible bands, where green
    substitutes for NIR and red remains as red. While not as accurate
    as true NDVI, it provides a similar normalized difference approach.
    
    Args:
        image: RGB image (numpy array, uint8)
    
    Returns:
        RGB-NDVI normalized to [0, 1]
    """
    # Convert to float for calculation
    img_float = image.astype(np.float32) / 255.0
    
    # Extract channels
    R = img_float[:, :, 0]
    G = img_float[:, :, 1]
    
    # Calculate RGB-NDVI with epsilon to avoid division by zero
    rgb_ndvi = (G - R) / (G + R + 1e-8)
    
    # Normalize to [0, 1] range
    rgb_ndvi_normalized = (rgb_ndvi - rgb_ndvi.min()) / (rgb_ndvi.max() - rgb_ndvi.min() + 1e-8)
    
    return rgb_ndvi_normalized


# ============================================================================
# ADAPTIVE THRESHOLDING AND CLASSIFICATION
# ============================================================================

def adaptive_threshold_percentile(vi_array, percentiles):
    """
    Apply adaptive thresholding based on data distribution percentiles.
    
    This approach is more robust than fixed thresholds as it adapts to
    the actual vegetation index distribution in each image, accounting
    for varying crop types, growth stages, and environmental conditions.
    
    Args:
        vi_array: Vegetation index array
        percentiles: Dictionary with 'poor' and 'moderate' percentile values
    
    Returns:
        Dictionary with threshold values
    """
    # Flatten the array for percentile calculation
    vi_flat = vi_array.flatten()
    
    # Calculate adaptive thresholds
    threshold_poor = np.percentile(vi_flat, percentiles['poor'])
    threshold_moderate = np.percentile(vi_flat, percentiles['moderate'])
    
    return {
        'poor': threshold_poor,
        'moderate': threshold_moderate
    }


def classify_health_adaptive(vi_array, thresholds):
    """
    Classify pixels into health categories using adaptive thresholds.
    
    Args:
        vi_array: Vegetation index array
        thresholds: Dictionary with 'poor' and 'moderate' threshold values
    
    Returns:
        health_map: Array with values 0 (poor), 1 (moderate), 2 (healthy)
    """
    health_map = np.zeros_like(vi_array, dtype=np.uint8)
    
    # Classify based on thresholds
    health_map[vi_array <= thresholds['poor']] = 0          # Poor
    health_map[(vi_array > thresholds['poor']) & 
               (vi_array <= thresholds['moderate'])] = 1    # Moderate
    health_map[vi_array > thresholds['moderate']] = 2       # Healthy
    
    return health_map


def kmeans_classification(vi_array, n_clusters=3):
    """
    Alternative classification using K-Means clustering.
    
    This unsupervised approach automatically identifies natural groupings
    in the vegetation index data, which can be useful when the health
    distribution doesn't follow expected patterns.
    
    Args:
        vi_array: Vegetation index array
        n_clusters: Number of health classes (default: 3)
    
    Returns:
        health_map: Clustered health classification
        cluster_centers: Centers of each cluster
    """
    # Reshape for K-Means
    vi_flat = vi_array.reshape(-1, 1)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(vi_flat)
    
    # Reshape back to image dimensions
    health_map = labels.reshape(vi_array.shape)
    
    # Sort clusters by their mean VI value (ascending)
    # So that: 0 = poor (lowest VI), 1 = moderate, 2 = healthy (highest VI)
    cluster_means = []
    for i in range(n_clusters):
        cluster_means.append((i, np.mean(vi_array[health_map == i])))
    
    cluster_means.sort(key=lambda x: x[1])
    
    # Remap labels to ordered classification
    remapped = np.zeros_like(health_map)
    for new_label, (old_label, _) in enumerate(cluster_means):
        remapped[health_map == old_label] = new_label
    
    return remapped.astype(np.uint8), kmeans.cluster_centers_


# ============================================================================
# VISUALIZATION AND OVERLAY
# ============================================================================

def create_health_overlay(original_image, health_map, alpha=0.5):
    """
    Create a color-coded health map overlay on the original image.
    
    Args:
        original_image: Original RGB image
        health_map: Health classification map (0=poor, 1=moderate, 2=healthy)
        alpha: Transparency factor for overlay (0=transparent, 1=opaque)
    
    Returns:
        Overlaid image
    """
    # Create color overlay
    overlay = np.zeros_like(original_image)
    
    # Apply colors based on health classification
    overlay[health_map == 0] = HEALTH_COLORS['poor']        # Red for poor
    overlay[health_map == 1] = HEALTH_COLORS['moderate']    # Orange for moderate
    overlay[health_map == 2] = HEALTH_COLORS['healthy']     # Green for healthy
    
    # Blend original image with overlay
    blended = cv2.addWeighted(original_image, 1 - alpha, overlay, alpha, 0)
    
    return blended


def create_health_heatmap(health_map):
    """
    Create a pure heatmap visualization of crop health.
    
    Args:
        health_map: Health classification map
    
    Returns:
        Color-coded heatmap image
    """
    # Create RGB heatmap
    heatmap = np.zeros((*health_map.shape, 3), dtype=np.uint8)
    
    heatmap[health_map == 0] = HEALTH_COLORS['poor']
    heatmap[health_map == 1] = HEALTH_COLORS['moderate']
    heatmap[health_map == 2] = HEALTH_COLORS['healthy']
    
    return heatmap


# ============================================================================
# SPATIAL ANALYSIS AND YIELD ESTIMATION
# ============================================================================

def calculate_health_statistics(health_map, pixel_area_m2=0.01):
    """
    Calculate area-based crop health statistics.
    
    Args:
        health_map: Health classification map
        pixel_area_m2: Area represented by each pixel in square meters
    
    Returns:
        Dictionary with health statistics
    """
    total_pixels = health_map.size
    
    # Count pixels in each category
    poor_pixels = np.sum(health_map == 0)
    moderate_pixels = np.sum(health_map == 1)
    healthy_pixels = np.sum(health_map == 2)
    
    # Calculate percentages
    poor_pct = (poor_pixels / total_pixels) * 100
    moderate_pct = (moderate_pixels / total_pixels) * 100
    healthy_pct = (healthy_pixels / total_pixels) * 100
    
    # Calculate areas
    poor_area = poor_pixels * pixel_area_m2
    moderate_area = moderate_pixels * pixel_area_m2
    healthy_area = healthy_pixels * pixel_area_m2
    total_area = total_pixels * pixel_area_m2
    
    return {
        'total_pixels': total_pixels,
        'poor_pixels': poor_pixels,
        'moderate_pixels': moderate_pixels,
        'healthy_pixels': healthy_pixels,
        'poor_percentage': poor_pct,
        'moderate_percentage': moderate_pct,
        'healthy_percentage': healthy_pct,
        'poor_area_m2': poor_area,
        'moderate_area_m2': moderate_area,
        'healthy_area_m2': healthy_area,
        'total_area_m2': total_area
    }


def estimate_yield(health_stats, yield_coefficients):
    """
    Estimate crop yield based on health distribution and area.
    
    This is a simplified model assuming:
    - Different productivity rates for each health category
    - Linear relationship between health and yield
    
    For academic purposes, coefficients should be calibrated with
    ground truth data for specific crop types.
    
    Args:
        health_stats: Dictionary from calculate_health_statistics()
        yield_coefficients: Dictionary with yield rates (kg/m²) per health class
    
    Returns:
        Dictionary with yield estimates
    """
    # Calculate yield for each health category
    poor_yield = health_stats['poor_area_m2'] * yield_coefficients['poor']
    moderate_yield = health_stats['moderate_area_m2'] * yield_coefficients['moderate']
    healthy_yield = health_stats['healthy_area_m2'] * yield_coefficients['healthy']
    
    # Total estimated yield
    total_yield_kg = poor_yield + moderate_yield + healthy_yield
    
    # Convert to tons
    total_yield_tons = total_yield_kg / 1000
    
    # Calculate yield per hectare (10,000 m²)
    total_area_ha = health_stats['total_area_m2'] / 10000
    yield_per_ha = total_yield_tons / total_area_ha if total_area_ha > 0 else 0
    
    return {
        'poor_yield_kg': poor_yield,
        'moderate_yield_kg': moderate_yield,
        'healthy_yield_kg': healthy_yield,
        'total_yield_kg': total_yield_kg,
        'total_yield_tons': total_yield_tons,
        'total_area_hectares': total_area_ha,
        'yield_per_hectare_tons': yield_per_ha
    }


# ============================================================================
# STREAMLIT APPLICATION
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="UAV Yield Monitoring System",
        page_icon=" ",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2E7D32;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header"> UAV Crop Health & Yield Monitoring</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Advanced RGB-based Vegetation Analysis System</p>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header(" Configuration")
    
    # Vegetation index selection
    vi_method = st.sidebar.selectbox(
        "Vegetation Index",
        ["ExG (Excess Green)", "VARI (Visible Atmospherically Resistant)", 
         "GLI (Green Leaf Index)", "RGB-NDVI"],
        help="Select the vegetation index method for analysis"
    )
    
    # Classification method
    classification_method = st.sidebar.selectbox(
        "Classification Method",
        ["Adaptive Percentile", "K-Means Clustering"],
        help="Method for classifying crop health levels"
    )
    
    # Overlay transparency
    overlay_alpha = st.sidebar.slider(
        "Overlay Transparency",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Adjust the transparency of the health overlay"
    )
    
    # Pixel area configuration
    pixel_area = st.sidebar.number_input(
        "Pixel Area (m²)",
        min_value=0.001,
        max_value=1.0,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Area represented by each pixel (depends on UAV altitude and camera)"
    )
    
    # Advanced settings expander
    with st.sidebar.expander("🔬 Advanced Settings"):
        if classification_method == "Adaptive Percentile":
            poor_percentile = st.slider(
                "Poor Health Threshold (%)",
                min_value=10,
                max_value=50,
                value=33,
                help="Percentile threshold for poor health classification"
            )
            moderate_percentile = st.slider(
                "Moderate Health Threshold (%)",
                min_value=50,
                max_value=90,
                value=67,
                help="Percentile threshold for moderate health classification"
            )
            custom_percentiles = {
                'poor': poor_percentile,
                'moderate': moderate_percentile
            }
        else:
            custom_percentiles = HEALTH_PERCENTILES
        
        # Yield coefficient customization
        st.subheader("Yield Coefficients (kg/m²)")
        healthy_coeff = st.number_input("Healthy", value=0.65, step=0.05, format="%.2f")
        moderate_coeff = st.number_input("Moderate", value=0.40, step=0.05, format="%.2f")
        poor_coeff = st.number_input("Poor", value=0.15, step=0.05, format="%.2f")
        
        custom_coefficients = {
            'healthy': healthy_coeff,
            'moderate': moderate_coeff,
            'poor': poor_coeff
        }
    
    # File uploader
    st.sidebar.markdown("---")
    uploaded_file = st.sidebar.file_uploader(
        "Upload UAV Image",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an RGB image from your UAV"
    )
    
    # Information about the methodology
    with st.sidebar.expander(" About This System"):
        st.markdown("""
        **Vegetation Indices:**
        - **ExG**: Emphasizes green vegetation, good for early-stage crops
        - **VARI**: Resistant to atmospheric effects, robust across lighting conditions
        - **GLI**: Normalized ExG, better contrast and illumination invariance
        - **RGB-NDVI**: Pseudo-NDVI using green as NIR substitute
        
        **Why Better Than Simple Green Averaging:**
        - Accounts for soil background and shadows
        - Normalizes for illumination variations
        - Uses spectral relationships (not just intensity)
        - Provides better discrimination between health levels
        
        **Academic Note:**
        This is a prototype demonstration system. For production deployment,
        coefficients should be calibrated with ground truth data for specific
        crop types and regional conditions.
        """)
    
    # Main content area
    if uploaded_file is not None:
        # Load and process image
        try:
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image_original = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image_original is None:
                st.error("❌ Error loading image. Please upload a valid image file.")
                return
            
            # Get original dimensions
            orig_height, orig_width = image_original.shape[:2]
            
            # Resize if needed to prevent browser crashes
            image, scale_factor = resize_image_if_needed(image_original)
            
            # Show info if resized
            if scale_factor < 1.0:
                st.info(f" Image resized for processing: {orig_width}x{orig_height} → {image.shape[1]}x{image.shape[0]} pixels (scale: {scale_factor:.2f})")
            
            # Convert BGR to RGB for display
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Display original image
            st.subheader("📷 Original UAV Image")
            st.image(image_rgb, use_container_width=True)
            
            # Calculate vegetation index based on selection
            with st.spinner(f'Calculating {vi_method}...'):
                if "ExG" in vi_method:
                    vi_array = calculate_exg(image_rgb)
                elif "VARI" in vi_method:
                    vi_array = calculate_vari(image_rgb)
                elif "GLI" in vi_method:
                    vi_array = calculate_gli(image_rgb)
                else:  # RGB-NDVI
                    vi_array = calculate_rgb_ndvi(image_rgb)
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(" Vegetation Index Map")
                # Display VI as heatmap
                vi_display = (vi_array * 255).astype(np.uint8)
                vi_colored = cv2.applyColorMap(vi_display, cv2.COLORMAP_JET)
                vi_colored_rgb = cv2.cvtColor(vi_colored, cv2.COLOR_BGR2RGB)
                st.image(vi_colored_rgb, use_container_width=True)
                
                # VI statistics
                st.metric("Mean VI Value", f"{np.mean(vi_array):.3f}")
                st.metric("Std Dev", f"{np.std(vi_array):.3f}")
            
            with col2:
                st.subheader(" VI Distribution")
                # Create histogram
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=vi_array.flatten(),
                    nbinsx=50,
                    marker_color='green',
                    opacity=0.7
                ))
                fig_hist.update_layout(
                    xaxis_title="Vegetation Index Value",
                    yaxis_title="Frequency",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Classification
            st.markdown("---")
            st.subheader(" Health Classification")
            
            with st.spinner('Classifying crop health...'):
                if classification_method == "Adaptive Percentile":
                    thresholds = adaptive_threshold_percentile(vi_array, custom_percentiles)
                    health_map = classify_health_adaptive(vi_array, thresholds)
                    
                    # Display thresholds
                    st.info(f"**Adaptive Thresholds:** Poor ≤ {thresholds['poor']:.3f} | "
                           f"Moderate ≤ {thresholds['moderate']:.3f} | "
                           f"Healthy > {thresholds['moderate']:.3f}")
                else:
                    health_map, cluster_centers = kmeans_classification(vi_array, n_clusters=3)
                    st.info(f"**K-Means Clustering:** {len(cluster_centers)} clusters identified")
            
            # Create visualizations
            overlay_image = create_health_overlay(image, health_map, alpha=overlay_alpha)
            overlay_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            
            heatmap_image = create_health_heatmap(health_map)
            heatmap_rgb = cv2.cvtColor(heatmap_image, cv2.COLOR_BGR2RGB)
            
            # Display health maps
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader(" Health Overlay")
                st.image(overlay_rgb, use_container_width=True)
            
            with col4:
                st.subheader(" Health Heatmap")
                st.image(heatmap_rgb, use_container_width=True)
            
            # Calculate statistics (adjust pixel area if image was resized)
            adjusted_pixel_area = pixel_area / (scale_factor ** 2) if scale_factor < 1.0 else pixel_area
            health_stats = calculate_health_statistics(health_map, pixel_area_m2=adjusted_pixel_area)
            
            # Display health statistics
            st.markdown("---")
            st.subheader("📈 Health Distribution Analysis")
            
            # Create three columns for metrics
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.metric(
                    label="🟢 Healthy",
                    value=f"{health_stats['healthy_percentage']:.1f}%",
                    delta=f"{health_stats['healthy_area_m2']:.1f} m²"
                )
            
            with metric_col2:
                st.metric(
                    label="🟠 Moderate",
                    value=f"{health_stats['moderate_percentage']:.1f}%",
                    delta=f"{health_stats['moderate_area_m2']:.1f} m²"
                )
            
            with metric_col3:
                st.metric(
                    label="🔴 Poor",
                    value=f"{health_stats['poor_percentage']:.1f}%",
                    delta=f"{health_stats['poor_area_m2']:.1f} m²"
                )
            
            # Pie chart for health distribution
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Healthy', 'Moderate', 'Poor'],
                values=[
                    health_stats['healthy_percentage'],
                    health_stats['moderate_percentage'],
                    health_stats['poor_percentage']
                ],
                marker=dict(colors=['#00FF00', '#FFA500', '#FF0000']),
                hole=0.3
            )])
            fig_pie.update_layout(
                title="Health Distribution",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Yield estimation
            st.markdown("---")
            st.subheader(" Yield Estimation")
            
            yield_estimates = estimate_yield(health_stats, custom_coefficients)
            
            # Display yield metrics
            yield_col1, yield_col2, yield_col3, yield_col4 = st.columns(4)
            
            with yield_col1:
                st.metric(
                    label="Total Yield",
                    value=f"{yield_estimates['total_yield_kg']:.1f} kg"
                )
            
            with yield_col2:
                st.metric(
                    label="Total Yield",
                    value=f"{yield_estimates['total_yield_tons']:.3f} tons"
                )
            
            with yield_col3:
                st.metric(
                    label="Total Area",
                    value=f"{yield_estimates['total_area_hectares']:.3f} ha"
                )
            
            with yield_col4:
                st.metric(
                    label="Yield per Hectare",
                    value=f"{yield_estimates['yield_per_hectare_tons']:.2f} t/ha"
                )
            
            # Yield breakdown by health category
            st.subheader(" Yield Breakdown by Health Category")
            
            yield_breakdown_df = pd.DataFrame({
                'Health Category': ['Healthy', 'Moderate', 'Poor'],
                'Area (m²)': [
                    health_stats['healthy_area_m2'],
                    health_stats['moderate_area_m2'],
                    health_stats['poor_area_m2']
                ],
                'Coefficient (kg/m²)': [
                    custom_coefficients['healthy'],
                    custom_coefficients['moderate'],
                    custom_coefficients['poor']
                ],
                'Yield (kg)': [
                    yield_estimates['healthy_yield_kg'],
                    yield_estimates['moderate_yield_kg'],
                    yield_estimates['poor_yield_kg']
                ]
            })
            
            st.dataframe(yield_breakdown_df, use_container_width=True)
            
            # Bar chart for yield contribution
            fig_bar = go.Figure(data=[
                go.Bar(
                    x=['Healthy', 'Moderate', 'Poor'],
                    y=[
                        yield_estimates['healthy_yield_kg'],
                        yield_estimates['moderate_yield_kg'],
                        yield_estimates['poor_yield_kg']
                    ],
                    marker=dict(color=['#00FF00', '#FFA500', '#FF0000'])
                )
            ])
            fig_bar.update_layout(
                title="Yield Contribution by Health Category",
                xaxis_title="Health Category",
                yaxis_title="Yield (kg)",
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Export functionality
            st.markdown("---")
            st.subheader(" Export Results")
            
            # Prepare report data
            report_data = {
                'Analysis Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Vegetation Index': vi_method,
                'Classification Method': classification_method,
                'Total Area (m²)': health_stats['total_area_m2'],
                'Total Area (ha)': yield_estimates['total_area_hectares'],
                'Healthy %': health_stats['healthy_percentage'],
                'Moderate %': health_stats['moderate_percentage'],
                'Poor %': health_stats['poor_percentage'],
                'Total Yield (kg)': yield_estimates['total_yield_kg'],
                'Total Yield (tons)': yield_estimates['total_yield_tons'],
                'Yield per Hectare (t/ha)': yield_estimates['yield_per_hectare_tons']
            }
            
            report_df = pd.DataFrame([report_data])
            
            # CSV download
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Report (CSV)",
                data=csv,
                file_name=f"crop_health_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Download processed images
            col_download1, col_download2 = st.columns(2)
            
            with col_download1:
                # Save overlay image with compression
                overlay_pil = Image.fromarray(overlay_rgb)
                buf_overlay = io.BytesIO()
                overlay_pil.save(buf_overlay, format="JPEG", quality=85, optimize=True)
                byte_overlay = buf_overlay.getvalue()
                
                st.download_button(
                    label="📥 Download Health Overlay",
                    data=byte_overlay,
                    file_name="health_overlay.jpg",
                    mime="image/jpeg"
                )
            
            with col_download2:
                # Save heatmap image with compression
                heatmap_pil = Image.fromarray(heatmap_rgb)
                buf_heatmap = io.BytesIO()
                heatmap_pil.save(buf_heatmap, format="JPEG", quality=85, optimize=True)
                byte_heatmap = buf_heatmap.getvalue()
                
                st.download_button(
                    label="📥 Download Health Heatmap",
                    data=byte_heatmap,
                    file_name="health_heatmap.jpg",
                    mime="image/jpeg"
                )
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)
    
    else:
        # Welcome screen with demo information
        st.info(" Please upload a UAV image using the sidebar to begin analysis")
        
        # Display methodology explanation
        st.markdown("---")
        st.subheader(" Methodology Overview")
        
        method_col1, method_col2 = st.columns(2)
        
        with method_col1:
            st.markdown("""
            **Vegetation Indices Used:**
            
            1. **ExG (Excess Green Index)**
               - Formula: `2*G - R - B`
               - Emphasizes green vegetation
               - Effective for early-stage crops
            
            2. **VARI (Visible Atmospherically Resistant Index)**
               - Formula: `(G - R) / (G + R - B)`
               - Robust to atmospheric effects
               - Good for varying lighting conditions
            
            3. **GLI (Green Leaf Index)**
               - Formula: `(2*G - R - B) / (2*G + R + B)`
               - Normalized version of ExG
               - Better illumination invariance
            
            4. **RGB-NDVI (Pseudo-NDVI)**
               - Formula: `(G - R) / (G + R)`
               - Green substitutes for NIR
               - Similar to traditional NDVI structure
            """)
        
        with method_col2:
            st.markdown("""
            **Why Better Than Simple Green Averaging?**
            
            ✅ **Spectral Relationships**: Uses ratios and differences between bands,
            not just raw intensities, which provides better discrimination
            
            ✅ **Normalization**: Reduces sensitivity to illumination variations,
            shadows, and atmospheric conditions
            
            ✅ **Soil Background Removal**: Emphasizes vegetation by contrasting
            green against red/blue channels
            
            ✅ **Adaptive Thresholding**: Uses percentile-based or clustering methods
            that adapt to each image's unique characteristics
            
            ✅ **Spatial Analysis**: Considers entire field distribution patterns,
            not just individual pixel values
            
            ✅ **Multi-Index Options**: Different indices perform better for
            different crops, growth stages, and conditions
            """)
        
        st.markdown("---")
        st.subheader("🎓 Academic Context")
        st.markdown("""
        This application is a **prototype demonstration system** for academic evaluation.
        
        **Key Features:**
        - RGB-only analysis (no NIR band required)
        - Multiple vegetation indices with theoretical foundations
        - Adaptive classification methods (percentile-based and K-Means)
        - Spatial analysis for area-based statistics
        - Yield estimation framework (requires calibration for production use)
        
        **Limitations & Future Work:**
        - Yield coefficients are demonstration values and should be calibrated with ground truth data
        - Assumes uniform crop type across the image
        - Does not account for plant phenology or growth stages
        - Simplified model without considering weather, soil quality, or management practices
        
        **For Production Deployment:**
        - Calibrate coefficients with actual harvest data
        - Implement crop-specific models
        - Add temporal analysis (multi-date imagery)
        - Integrate meteorological data
        - Validate against ground-based measurements
        """)


if __name__ == "__main__":
    main()
