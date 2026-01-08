# Chapter 2: Hyperspectral Imaging Fundamentals

## Learning Objectives
 
By the end of this chapter, students will be able to:
- Understand the principles of hyperspectral imaging
- Explain the electromagnetic spectrum and its relevance to vegetation
- Identify key vegetation indices used in crop health assessment
- Describe how hyperspectral data is collected and processed
- Compare hyperspectral imaging with other remote sensing techniques

## Key Concepts

### Understanding the Electromagnetic Spectrum

The electromagnetic spectrum encompasses all types of electromagnetic radiation, from radio waves to gamma rays. For agricultural applications, we focus on the visible and infrared portions:

```
Electromagnetic Spectrum (Relevant Portions)
0.4 μm     0.7 μm      1.3 μm      1.6 μm      2.5 μm
|----------|-----------|-----------|-----------|
Visible    Near IR     Shortwave   Shortwave
           (NIR)       IR-1        IR-2
```

**Key Wavelength Regions**:
- **Visible (0.4-0.7 μm)**: Blue (0.4-0.5 μm), Green (0.5-0.6 μm), Red (0.6-0.7 μm)
- **Near Infrared (NIR) (0.7-1.3 μm)**: Strong reflectance from healthy vegetation
- **Shortwave Infrared (SWIR) (1.3-2.5 μm)**: Water absorption features

### Principles of Hyperspectral Imaging

Hyperspectral imaging captures and processes information across the electromagnetic spectrum, typically collecting hundreds of narrow spectral bands. This differs from:

- **Panchromatic Imaging**: Single broad band (black and white)
- **Multispectral Imaging**: Few discrete bands (typically 3-10)
- **Hyperspectral Imaging**: Many narrow contiguous bands (typically 100+)

**Advantages of Hyperspectral Imaging**:
- High spectral resolution
- Detailed spectral signatures
- Ability to detect subtle differences
- Quantitative analysis capabilities

### Spectral Signatures of Vegetation

Healthy vegetation has a characteristic spectral signature:

1. **Visible Region**: Low reflectance due to chlorophyll absorption
   - Blue and red: Strong absorption
   - Green: Moderate absorption (hence the green color)

2. **Red Edge (0.7-0.75 μm)**: Rapid increase in reflectance
   - Transition from strong absorption to strong reflection
   - Sensitive to chlorophyll content

3. **Near Infrared (0.7-1.3 μm)**: High reflectance
   - Due to internal leaf structure
   - Strong indicator of plant health

4. **Shortwave Infrared (1.3-2.5 μm)**: Water absorption features
   - Sensitive to water content
   - Useful for stress detection

### Vegetation Indices

Vegetation indices are mathematical combinations of spectral bands that highlight vegetation characteristics.

#### Common Vegetation Indices

1. **Normalized Difference Vegetation Index (NDVI)**
   ```
   NDVI = (NIR - Red) / (NIR + Red)
   Range: -1 to 1
   Healthy vegetation: 0.7-1.0
   ```

2. **Enhanced Vegetation Index (EVI)**
   ```
   EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
   More sensitive to canopy variations
   ```

3. **Soil Adjusted Vegetation Index (SAVI)**
   ```
   SAVI = (NIR - Red) / (NIR + Red + L) * (1 + L)
   Where L is a soil adjustment factor
   ```

4. **Green Normalized Difference Vegetation Index (GNDVI)**
   ```
   GNDVI = (NIR - Green) / (NIR + Green)
   Sensitive to chlorophyll content
   ```

### Data Collection Methods

#### Satellite-Based Systems
- **Commercial Satellites**: WorldView, QuickBird
- **Research Satellites**: Hyperion (EO-1), PRISMA
- **Advantages**: Large area coverage, regular revisits
- **Limitations**: Lower spatial resolution, less frequent revisits

#### Airborne Systems
- **UAV/Drone Systems**: Custom hyperspectral sensors
- **Airborne Sensors**: AVIRIS, HyMap
- **Advantages**: Higher resolution, flexible scheduling
- **Limitations**: Weather dependent, operational costs

#### Ground-Based Systems
- **Handheld Spectrometers**: For spot measurements
- **Tractor-mounted Sensors**: For field-scale mapping
- **Research-grade Instruments**: ASD FieldSpec, GER series
- **Advantages**: High resolution, controlled conditions
- **Limitations**: Limited coverage, time-consuming

## Hyperspectral Data Processing

### Data Preprocessing Steps

1. **Radiometric Calibration**: Converting digital numbers to radiance or reflectance
2. **Atmospheric Correction**: Removing atmospheric effects
3. **Geometric Correction**: Correcting for sensor and platform geometry
4. **Noise Reduction**: Filtering out sensor and environmental noise

### Spectral Analysis Techniques

#### Spectral Signature Analysis
Comparing observed spectra with reference spectra to identify materials:

```python
# Example of spectral signature comparison
def compare_spectral_signatures(observed, reference):
    # Calculate spectral angle mapper (SAM)
    dot_product = np.dot(observed, reference)
    norms = np.linalg.norm(observed) * np.linalg.norm(reference)
    sam = np.arccos(dot_product / norms)
    return sam
```

#### Spectral Feature Extraction
Identifying specific absorption features:

```python
# Example of red edge detection
def detect_red_edge(spectrum, wavelengths):
    # Find the red edge position (maximum of first derivative)
    derivative = np.gradient(spectrum, wavelengths)
    red_edge_pos = wavelengths[np.argmax(derivative)]
    return red_edge_pos
```

## Applications in Crop Health Monitoring

### Stress Detection

Hyperspectral imaging can detect various types of crop stress:

1. **Water Stress**: Changes in water absorption features (1.45 μm, 1.95 μm)
2. **Nutrient Deficiency**: Altered chlorophyll absorption (red, blue)
3. **Disease**: Changes in cell structure (NIR reflectance)
4. **Pest Damage**: Physical damage to leaf structure

### Disease Identification

Different plant diseases cause distinct spectral changes:

- **Fungal Diseases**: Generally reduce NIR reflectance
- **Bacterial Diseases**: May increase visible reflectance
- **Viral Infections**: Often cause mottling patterns

### Growth Stage Monitoring

Hyperspectral data can track crop development:

- **Germination**: Low vegetation indices
- **Vegetative Growth**: Increasing indices
- **Flowering**: Peak indices
- **Senescence**: Declining indices

## Data Formats and Storage

### Common Data Formats

1. **Binary Formats**: .npy, .mat for efficient storage
2. **Image Formats**: .tif with metadata
3. **Text Formats**: .csv for simple data exchange

### Data Structure

Hyperspectral data is typically organized as 3D arrays:
- **Dimensions**: [bands, height, width]
- **Example**: 224 bands × 128 pixels × 128 pixels

### Metadata Requirements

Essential metadata includes:
- Spectral calibration information
- Spatial coordinates
- Acquisition time
- Environmental conditions
- Processing history

## Practical Exercises

### Exercise 1: Analyzing Spectral Signatures

```python
import numpy as np
import matplotlib.pyplot as plt

# Create sample spectral signatures
wavelengths = np.linspace(400, 2500, 224)  # 224 bands from 400-2500 nm

# Healthy vegetation signature
healthy_veg = np.zeros_like(wavelengths)
healthy_veg[wavelengths < 700] = 0.05  # Low reflectance in visible
healthy_veg[(wavelengths >= 700) & (wavelengths <= 1300)] = 0.4  # High NIR
healthy_veg[wavelengths > 1300] = 0.3  # SWIR

# Stressed vegetation signature
stressed_veg = healthy_veg.copy()
stressed_veg[(wavelengths >= 700) & (wavelengths <= 1300)] *= 0.7  # Reduced NIR

# Plot comparison
plt.figure(figsize=(10, 6))
plt.plot(wavelengths, healthy_veg, label='Healthy Vegetation', linewidth=2)
plt.plot(wavelengths, stressed_veg, label='Stressed Vegetation', linewidth=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Reflectance')
plt.title('Spectral Signatures of Healthy vs Stressed Vegetation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Exercise 2: Calculating Vegetation Indices

```python
def calculate_ndvi(nir_band, red_band):
    """Calculate NDVI from NIR and Red bands"""
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-8)
    return ndvi

# Example with sample data
nir_data = np.random.normal(0.6, 0.1, (100, 100))  # Sample NIR data
red_data = np.random.normal(0.1, 0.05, (100, 100))  # Sample Red data

ndvi_map = calculate_ndvi(nir_data, red_data)
print(f"NDVI range: {ndvi_map.min():.3f} to {ndvi_map.max():.3f}")
print(f"Mean NDVI: {ndvi_map.mean():.3f}")
```

## Discussion Questions

1. How does the spectral signature of healthy vegetation differ from stressed vegetation?
2. What are the advantages and disadvantages of satellite-based vs. drone-based hyperspectral imaging?
3. Why are vegetation indices useful for crop health monitoring?
4. How can hyperspectral imaging detect different types of crop stress?

## Additional Resources

- Jensen, J.R. "Remote Sensing of the Environment"
- Thenkabail, P.S. "Hyperspectral Remote Sensing of Vegetation"
- NASA's Applied Sciences Program resources
- Research papers on agricultural hyperspectral applications

## Summary

This chapter covered the fundamentals of hyperspectral imaging and its application in crop health monitoring. We explored the electromagnetic spectrum, spectral signatures of vegetation, and key vegetation indices used in agricultural applications. We also discussed data collection methods and processing techniques. Understanding these concepts is crucial for working with the AI-powered spectral health mapping system, which we'll explore in more detail in the next chapter.