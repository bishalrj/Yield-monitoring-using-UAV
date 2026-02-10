# 🚀 Quick Start Guide

## Get Up and Running in 3 Minutes

### Step 1: Install Dependencies (1 minute)
```bash
pip install streamlit opencv-python numpy Pillow pandas plotly scikit-learn
```

### Step 2: Run the Application (30 seconds)
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

### Step 3: Test with Sample Images (1 minute)

#### Option A: Generate Test Images
```bash
python generate_test_images.py
```
This creates 4 synthetic crop field images in `test_images/` directory.

#### Option B: Use Your Own UAV Images
Upload any RGB aerial image of a crop field (JPG, PNG, BMP)

---

## 📸 First Analysis

1. **Upload Image**: Click "Upload UAV Image" in the sidebar
2. **Select Method**: Keep default settings (ExG + Adaptive Percentile)
3. **Click Analyze**: Results appear automatically
4. **Explore Results**: 
   - View vegetation index maps
   - See health classification overlay
   - Check statistics and yield estimates
5. **Export**: Download CSV report and images

---

## ⚙️ Key Settings to Adjust

### Pixel Area (Most Important!)
- **What it is**: Area covered by each pixel in m²
- **How to calculate**: (Camera GSD)²
- **Examples**:
  - 1cm GSD → 0.0001 m²
  - 5cm GSD → 0.0025 m²
  - 10cm GSD → 0.01 m² (default)
  - 20cm GSD → 0.04 m²

### Vegetation Index
- **ExG**: Best for green crops, early stages
- **VARI**: Best for varying conditions
- **GLI**: Good all-purpose index
- **RGB-NDVI**: Familiar NDVI structure

### Classification Method
- **Adaptive Percentile**: Recommended (adapts to each image)
- **K-Means**: For non-standard distributions

---

## 🎯 Expected Results

With the generated test images, you should see:

**mixed_health_field.jpg**:
- ~33% healthy (green zone at top)
- ~33% moderate (orange zone in middle)
- ~33% poor (red zone at bottom)

**healthy_field.jpg**:
- ~80-90% healthy classification
- High vegetation index values

**moderate_field.jpg**:
- ~60-70% moderate classification
- Medium vegetation index values

**poor_field.jpg**:
- ~60-70% poor classification
- Low vegetation index values

---

## 🔍 Troubleshooting

**App won't start?**
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Port already in use?**
```bash
streamlit run app.py --server.port 8502
```

**Can't find test images?**
```bash
# Make sure you're in the same directory as generate_test_images.py
ls test_images/
```

---

## 📊 Understanding the Output

### Vegetation Index Map
- **Blue/Purple**: Low vegetation (soil, stressed crops)
- **Green/Yellow**: Medium vegetation
- **Red/White**: High vegetation (healthy crops)

### Health Overlay
- **🟢 Green**: Healthy crops (top 33%)
- **🟠 Orange**: Moderate health (middle 33%)
- **🔴 Red**: Poor health (bottom 33%)

### Statistics
- **Percentages**: Proportion of field in each health category
- **Areas**: Actual m² coverage
- **Yield**: Estimated production in kg/tons

---

## 🎓 Academic Demo Tips

**For Presentations:**
1. Start with the mixed health field
2. Show how different indices compare
3. Demonstrate adaptive vs K-Means
4. Export and show the CSV report

**For Evaluation:**
- Explain why RGB indices work without NIR
- Discuss adaptive thresholding advantages
- Address limitations and assumptions
- Show methodology in sidebar info

**For Questions:**
- Refer to README.md technical details
- Check inline code comments
- Reference academic literature on vegetation indices

---

## 📈 Next Steps

Once you're comfortable with the basics:

1. **Try Real Data**: Upload actual UAV imagery
2. **Calibrate Coefficients**: Use ground truth to adjust yield parameters
3. **Compare Indices**: Test different VI methods on same image
4. **Customize Thresholds**: Adjust percentiles in advanced settings
5. **Export Results**: Build a report for your project

---

## 💡 Pro Tips

✅ **For Best Results:**
- Use nadir (straight-down) UAV images
- Ensure good, even lighting
- Avoid images with shadows or clouds
- Crop to vegetation-only areas if possible

✅ **For Demonstrations:**
- Generate all 4 test images beforehand
- Pre-configure settings for your crop type
- Have sample reports ready to show
- Prepare explanation of methodology

✅ **For Academic Projects:**
- Document your calibration process
- Include ground truth validation
- Compare with alternative methods
- Discuss limitations and future work

---

## 🆘 Need Help?

1. **Check README.md** - Comprehensive technical documentation
2. **Review Code Comments** - Every function is explained
3. **Sidebar Info** - Methodology overview in the app
4. **Test Images** - Use synthetic data to verify setup

---

## ✨ You're Ready!

That's it! You now have a fully functional crop health monitoring system.

**Enjoy exploring precision agriculture with computer vision! 🌾**
