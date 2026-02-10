# UAV Yield Monitoring System

**Academic Prototype for Crop Health and Yield Estimation using RGB Imagery**

This Streamlit application provides advanced crop health analysis and yield estimation using only RGB imagery from UAVs (no NIR band required). It implements multiple vegetation indices, adaptive thresholding, and spatial analysis techniques suitable for academic demonstrations and research prototypes.

---

## 🌟 Features

### Vegetation Indices
- **ExG (Excess Green Index)**: Emphasizes green vegetation for early-stage crop detection
- **VARI (Visible Atmospherically Resistant Index)**: Robust across varying lighting conditions
- **GLI (Green Leaf Index)**: Normalized approach with better illumination invariance
- **RGB-NDVI (Pseudo-NDVI)**: Green-substituted NDVI approximation

### Classification Methods
- **Adaptive Percentile Thresholding**: Data-driven thresholds based on image distribution
- **K-Means Clustering**: Unsupervised classification identifying natural groupings

### Analysis Outputs
- Color-coded health maps and overlays
- Area-based health statistics (healthy/moderate/poor percentages)
- Vegetation index heatmaps and distributions
- Yield estimation with customizable coefficients
- Comprehensive reporting and data export

---

## 📋 Requirements

- Python 3.8 or higher
- pip package manager

---

## 🚀 Installation

### Step 1: Clone or Download
Download the application files to your local directory.

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit opencv-python numpy Pillow pandas plotly scikit-learn
```

---

## ▶️ Running the Application

### Launch the App
```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`

### Alternative Port
If port 8501 is busy:
```bash
streamlit run app.py --server.port 8502
```

---

## 📖 Usage Guide

### 1. Upload Image
- Click **"Upload UAV Image"** in the sidebar
- Select an RGB image (JPG, PNG, BMP)
- Recommended: Nadir (top-down) UAV imagery of crop fields

### 2. Configure Analysis
**Vegetation Index**: Choose from ExG, VARI, GLI, or RGB-NDVI
- ExG: Best for early-stage crops with distinct green coloration
- VARI: Ideal for multi-temporal analysis with varying conditions
- GLI: Good general-purpose index with normalization
- RGB-NDVI: Familiar NDVI-like structure for comparison

**Classification Method**:
- **Adaptive Percentile**: Recommended for most cases; adapts to each image
- **K-Means Clustering**: Useful for images with non-standard distributions

### 3. Adjust Parameters
**Overlay Transparency**: Control visibility of health overlay (0.3-0.7 recommended)

**Pixel Area**: Set based on your UAV specifications
- Calculate as: `(GSD)²` where GSD = Ground Sample Distance in meters
- Example: 1cm GSD → 0.0001 m² per pixel
- Example: 10cm GSD → 0.01 m² per pixel

**Advanced Settings**:
- Customize percentile thresholds for health classification
- Adjust yield coefficients (kg/m²) for each health category

### 4. View Results
The application displays:
- **Original Image**: Your uploaded UAV imagery
- **Vegetation Index Map**: Heatmap showing VI distribution
- **Health Overlay**: Original image with color-coded health overlay
- **Health Heatmap**: Pure health classification visualization
- **Statistics**: Percentages and areas for each health category
- **Yield Estimates**: Total and per-hectare yield predictions

### 5. Export Data
- Download CSV report with all statistics
- Save health overlay and heatmap images
- Use data for further analysis or reporting

---

## 🧪 Technical Details

### Why These Indices are Better Than Simple Green Averaging

**1. Spectral Relationships**
- Simple averaging: `mean(G)`
- Our indices: `f(G, R, B)` using ratios and differences
- Captures vegetation characteristics, not just brightness

**2. Normalization**
- Reduces sensitivity to:
  - Sun angle variations
  - Shadow effects
  - Atmospheric haze
  - Camera exposure settings

**3. Soil Background Removal**
- Green averaging includes soil pixels
- Indices use spectral contrast to emphasize vegetation
- Better discrimination in sparse canopy conditions

**4. Adaptive Thresholding**
- Fixed thresholds fail across different:
  - Crop types
  - Growth stages
  - Soil types
  - Imaging conditions
- Our percentile-based approach adapts to each image's unique distribution

### Vegetation Index Formulas

**ExG (Excess Green)**
```
ExG = 2*G - R - B
```
- Range: Variable (normalized to [0,1])
- Best for: Early-stage crops, row crops
- Sensitivity: High to green pigmentation

**VARI (Visible Atmospherically Resistant Index)**
```
VARI = (G - R) / (G + R - B)
```
- Range: -1 to 1 (normalized to [0,1])
- Best for: Multi-temporal analysis, varying conditions
- Sensitivity: Resistant to atmospheric effects

**GLI (Green Leaf Index)**
```
GLI = (2*G - R - B) / (2*G + R + B)
```
- Range: -1 to 1 (normalized to [0,1])
- Best for: General purpose, mixed vegetation
- Sensitivity: Normalized ExG with better invariance

**RGB-NDVI**
```
RGB-NDVI = (G - R) / (G + R)
```
- Range: -1 to 1 (normalized to [0,1])
- Best for: Comparison with traditional NDVI
- Sensitivity: Similar structure to NIR-based NDVI

### Health Classification

**Adaptive Percentile Method**
```python
threshold_poor = percentile(VI, 33)      # Bottom 33%
threshold_moderate = percentile(VI, 67)  # Middle 33%
# Top 33% classified as healthy
```

**K-Means Clustering**
- Automatically identifies 3 clusters in VI distribution
- Sorts clusters by mean VI value
- Assigns health labels accordingly

### Yield Estimation Model

```
Yield_total = Σ(Area_i × Coefficient_i)

where:
  i ∈ {healthy, moderate, poor}
  Area_i = pixel_count_i × pixel_area_m²
  Coefficient_i = productivity_rate (kg/m²)
```

**Default Coefficients** (demonstration values):
- Healthy: 0.65 kg/m²
- Moderate: 0.40 kg/m²
- Poor: 0.15 kg/m²

⚠️ **Important**: These coefficients are illustrative. For actual yield prediction, calibrate with:
- Historical harvest data
- Crop-specific growth models
- Regional productivity averages
- Ground truth sampling

---

## 📊 Sample Workflow

### Example Analysis Process

1. **Capture**: UAV flies over field at 50m altitude with 12MP camera
   - GSD ≈ 2cm → Pixel area = 0.0004 m²

2. **Upload**: Load geo-referenced orthomosaic into application

3. **Configure**:
   - VI: VARI (consistent lighting across flight)
   - Classification: Adaptive Percentile
   - Pixel area: 0.0004 m²

4. **Analyze**: Application computes:
   - VARI map showing vegetation vigor
   - Health classification: 65% healthy, 25% moderate, 10% poor
   - Total area: 2.5 hectares
   - Estimated yield: 12.3 tons (4.9 t/ha)

5. **Export**: Save results for farm management decisions

---

## 🎓 Academic Context

### Intended Use
This application is designed as a **demonstration prototype** for academic projects, research presentations, and educational purposes. It showcases:
- Computer vision techniques for agriculture
- RGB-based vegetation analysis
- Spatial data analysis
- Interactive data visualization

### Limitations
- **Simplified Model**: Actual yield depends on many factors not considered here (weather, soil, pests, management)
- **Calibration Required**: Coefficients must be validated with ground truth data
- **Crop Agnostic**: Does not account for crop-specific characteristics
- **Static Analysis**: Single-image analysis without temporal context
- **No Phenology**: Doesn't consider growth stages

### Validation Steps for Production
1. **Ground Truth Collection**: Take field samples during harvest
2. **Coefficient Calibration**: Fit model to actual yield data
3. **Cross-Validation**: Test on independent fields
4. **Temporal Analysis**: Incorporate multi-date imagery
5. **Crop Models**: Integrate physiological growth models
6. **Environmental Data**: Add weather, soil, and management variables

### Academic References
For deeper understanding, consider reviewing:
- Spectral vegetation indices in precision agriculture
- Remote sensing for crop monitoring
- Machine learning in agricultural applications
- UAV-based crop health assessment methods

---

## 🔧 Troubleshooting

### Common Issues

**"Module not found" error**
```bash
pip install --upgrade -r requirements.txt
```

**Port already in use**
```bash
streamlit run app.py --server.port 8502
```

**Image won't upload**
- Check file format (JPG, PNG, BMP only)
- Ensure file size < 200MB
- Try converting image to RGB mode if grayscale

**Values seem incorrect**
- Verify pixel area calculation (critical for area/yield estimates)
- Check that image is aerial/nadir view (not oblique)
- Ensure proper lighting (avoid shadows, overexposure)

**K-Means produces strange results**
- Try Adaptive Percentile method instead
- Image may have non-vegetation dominant features
- Consider cropping to vegetation-only area

---

## 📝 Citation

If you use this application in your research or academic work, please acknowledge:

```
UAV Yield Monitoring System (2026)
RGB-based Crop Health and Yield Estimation Prototype
Academic Demonstration Application
```

---

## 📄 License

This is an academic prototype application. Feel free to use, modify, and extend for educational and research purposes.

---

## 🤝 Contributing

This is a demonstration prototype. For improvements:
- Add new vegetation indices (GRVI, NGRDI, etc.)
- Implement temporal analysis
- Integrate machine learning models
- Add multi-spectral support
- Enhance yield prediction models

---

## 📞 Support

For technical questions about the implementation:
- Review the inline code comments (comprehensive documentation)
- Check the methodology section in the sidebar
- Refer to academic literature on RGB-based vegetation indices

---

## 🔄 Version History

**v1.0 (February 2026)**
- Initial release
- 4 vegetation indices (ExG, VARI, GLI, RGB-NDVI)
- 2 classification methods
- Yield estimation framework
- Interactive Streamlit interface
- Export functionality

---

## 📚 Additional Resources

### Recommended Reading
- Vegetation indices in precision agriculture
- UAV-based crop monitoring techniques
- Computer vision for agricultural applications
- Spectral analysis of crop health

### Data Sources
For testing, you can use:
- Free agricultural UAV datasets (e.g., OpenAerialMap)
- Satellite RGB imagery downsampled to UAV resolution
- Synthetic crop field images
- Your own UAV captures

---

**Built with ❤️ for Agricultural Research**

*Empowering precision agriculture through accessible computer vision technology*
