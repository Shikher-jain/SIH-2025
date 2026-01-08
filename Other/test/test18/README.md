# Test18 - GeoJSON Area Processing Pipeline

A Python-based pipeline for processing satellite imagery and sensor data from multiple GeoJSON areas. This pipeline processes each area coordinate individually and generates combined data products.

## Features

- Processes multiple GeoJSON area coordinates one by one
- Minimizes pixel loss within GEE thresholds
- Outputs structured directory format: `test18_output/areaname/area_coord_index/{feature_id}_stacked.npy`, `{feature_id}_ndvi.png`, `{feature_id}_rgb.png`
- Uses date range October 2017 to January 2018
- Implements 50% cloud threshold
- Uses the same 5 sensors as test15
- Skips areas with no data or size issues without creating output folders

## Requirements

- Python 3.7+
- Google Earth Engine API
- NumPy
- Matplotlib

## Installation

1. Install the required Python packages:
```bash
pip install -r requirements.txt
```

2. Ensure you have the Earth Engine service account JSON file in the parent directory (`../earth-engine-service-account.json`)

## Configuration

The pipeline uses a JSON configuration file (`config.json`) with the following structure:

```json
{
  "area": {
    "name": "Tarn_Taran",
    "type": "geojson",
    "geojson": {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "id": "area_1"
          },
          "geometry": {
            "coordinates": [ /* polygon coordinates */ ],
            "type": "Polygon"
          }
        }
        // Additional features...
      ]
    }
  }
}
```

## Usage

Run the pipeline with the default configuration:
```bash
python main.py
```

Or specify a custom configuration file:
```bash
python main.py --config path/to/your/config.json
```

## Output Structure

The pipeline generates the following output structure:
```
test18_output/
└── areaname/
    ├── 1/
    │   ├── feature_id_stacked.npy
    │   ├── feature_id_ndvi.png
    │   └── feature_id_rgb.png
    ├── 2/
    │   ├── feature_id_stacked.npy
    │   ├── feature_id_ndvi.png
    │   └── feature_id_rgb.png
    └── ...
```

## Data Products

1. **Stacked Data (`feature_id_stacked.npy`)**: A 3D NumPy array combining NDVI and sensor data with dimensions (height, width, 21)
   - Band 0: NDVI
   - Bands 1-4: ECe (mean, median, std, max)
   - Bands 5-8: N (mean, median, std, max)
   - Bands 9-12: P (mean, median, std, max)
   - Bands 13-16: pH (mean, median, std, max)
   - Bands 17-20: OC (mean, median, std, max)

2. **NDVI Visualization (`feature_id_ndvi.png`)**: A PNG image of the NDVI data with a red-to-green colormap

3. **RGB Visualization (`feature_id_rgb.png`)**: A PNG image of the RGB composite from Sentinel-2 data

## Skipping Behavior

If an area cannot be processed due to:
- No suitable satellite imagery found
- Image size exceeding GEE limits
- Other data issues

The pipeline will skip that area and continue with the next one without creating an output folder for the skipped area.

## Logging

The pipeline provides detailed logging of its operations, including:
- Initialization status
- Processing progress
- Success/failure of each area
- Error messages for troubleshooting