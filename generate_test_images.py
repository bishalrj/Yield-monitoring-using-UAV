"""
Generate Synthetic Test Images for UAV Yield Monitoring Demo

This script creates realistic-looking crop field images for testing
the application without requiring actual UAV imagery.
"""

import cv2
import numpy as np
from PIL import Image
import os

def generate_crop_field_image(width=800, height=600, filename="test_crop_field.jpg"):
    """
    Generate a synthetic crop field image with varying health zones.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        filename: Output filename
    """
    # Create base soil layer (brown tones)
    soil_base = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
    soil_base[:, :, 0] = np.random.randint(60, 90, (height, width))   # Blue
    soil_base[:, :, 1] = np.random.randint(80, 110, (height, width))  # Green
    soil_base[:, :, 2] = np.random.randint(90, 130, (height, width))  # Red
    
    # Create vegetation mask with varying densities
    vegetation = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Define three health zones
    zone_height = height // 3
    
    # Healthy zone (top third) - vibrant green
    healthy_mask = np.random.rand(zone_height, width) > 0.2
    vegetation[0:zone_height, :, 0][healthy_mask] = np.random.randint(20, 60, np.sum(healthy_mask))   # Blue
    vegetation[0:zone_height, :, 1][healthy_mask] = np.random.randint(100, 160, np.sum(healthy_mask)) # Green
    vegetation[0:zone_height, :, 2][healthy_mask] = np.random.randint(30, 70, np.sum(healthy_mask))   # Red
    
    # Moderate zone (middle third) - yellowish green
    moderate_mask = np.random.rand(zone_height, width) > 0.3
    vegetation[zone_height:2*zone_height, :, 0][moderate_mask] = np.random.randint(40, 80, np.sum(moderate_mask))   # Blue
    vegetation[zone_height:2*zone_height, :, 1][moderate_mask] = np.random.randint(90, 130, np.sum(moderate_mask))  # Green
    vegetation[zone_height:2*zone_height, :, 2][moderate_mask] = np.random.randint(50, 100, np.sum(moderate_mask))  # Red
    
    # Poor zone (bottom third) - sparse, brownish
    poor_mask = np.random.rand(zone_height, width) > 0.5
    vegetation[2*zone_height:, :, 0][poor_mask] = np.random.randint(60, 100, np.sum(poor_mask))  # Blue
    vegetation[2*zone_height:, :, 1][poor_mask] = np.random.randint(70, 110, np.sum(poor_mask))  # Green
    vegetation[2*zone_height:, :, 2][poor_mask] = np.random.randint(70, 120, np.sum(poor_mask))  # Red
    
    # Blend soil and vegetation
    veg_weight = (vegetation.sum(axis=2, keepdims=True) > 0).astype(np.float32)
    blended = (soil_base.astype(np.float32) * (1 - veg_weight * 0.7) + 
               vegetation.astype(np.float32) * 0.7).astype(np.uint8)
    
    # Add texture with Gaussian blur
    blended = cv2.GaussianBlur(blended, (5, 5), 0)
    
    # Add some random variations for realism
    noise = np.random.randint(-10, 10, blended.shape, dtype=np.int16)
    blended = np.clip(blended.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add subtle row patterns (optional - simulates crop rows)
    for i in range(0, width, 20):
        blended[:, i:i+2] = (blended[:, i:i+2] * 0.9).astype(np.uint8)
    
    # Save image
    cv2.imwrite(filename, blended)
    print(f"✅ Generated test image: {filename}")
    print(f"   Size: {width}x{height} pixels")
    print(f"   Zones: Healthy (top), Moderate (middle), Poor (bottom)")
    
    return blended


def generate_uniform_field(width=800, height=600, health_level="healthy", 
                          filename="test_uniform_field.jpg"):
    """
    Generate a uniform crop field with consistent health.
    
    Args:
        width: Image width
        height: Image height
        health_level: 'healthy', 'moderate', or 'poor'
        filename: Output filename
    """
    # Base soil
    soil = np.random.randint(80, 120, (height, width, 3), dtype=np.uint8)
    
    # Define vegetation colors based on health
    if health_level == "healthy":
        coverage = 0.85
        b_range, g_range, r_range = (20, 50), (110, 160), (30, 60)
    elif health_level == "moderate":
        coverage = 0.65
        b_range, g_range, r_range = (40, 70), (90, 130), (50, 90)
    else:  # poor
        coverage = 0.45
        b_range, g_range, r_range = (60, 90), (70, 110), (70, 110)
    
    # Create vegetation
    vegetation = np.zeros((height, width, 3), dtype=np.uint8)
    mask = np.random.rand(height, width) > (1 - coverage)
    
    vegetation[:, :, 0][mask] = np.random.randint(*b_range, np.sum(mask))
    vegetation[:, :, 1][mask] = np.random.randint(*g_range, np.sum(mask))
    vegetation[:, :, 2][mask] = np.random.randint(*r_range, np.sum(mask))
    
    # Blend
    veg_weight = (vegetation.sum(axis=2, keepdims=True) > 0).astype(np.float32)
    blended = (soil.astype(np.float32) * (1 - veg_weight * 0.7) + 
               vegetation.astype(np.float32) * 0.7).astype(np.uint8)
    
    # Add texture
    blended = cv2.GaussianBlur(blended, (5, 5), 0)
    
    # Save
    cv2.imwrite(filename, blended)
    print(f"✅ Generated uniform {health_level} field: {filename}")
    
    return blended


if __name__ == "__main__":
    print("🌾 Generating synthetic crop field test images...\n")
    
    # Create output directory
    os.makedirs("test_images", exist_ok=True)
    
    # Generate various test images
    print("1. Mixed health zones field:")
    generate_crop_field_image(
        width=800, 
        height=600, 
        filename="test_images/mixed_health_field.jpg"
    )
    
    print("\n2. Healthy uniform field:")
    generate_uniform_field(
        width=800,
        height=600,
        health_level="healthy",
        filename="test_images/healthy_field.jpg"
    )
    
    print("\n3. Moderate uniform field:")
    generate_uniform_field(
        width=800,
        height=600,
        health_level="moderate",
        filename="test_images/moderate_field.jpg"
    )
    
    print("\n4. Poor uniform field:")
    generate_uniform_field(
        width=800,
        height=600,
        health_level="poor",
        filename="test_images/poor_field.jpg"
    )
    
    print("\n✨ Test image generation complete!")
    print("📁 Images saved in 'test_images/' directory")
    print("\n🚀 You can now upload these images in the Streamlit app for testing.")
