# Spectral Health Mapping API

A comprehensive FastAPI application for crop health analysis using hyperspectral imaging and machine learning.

## Features

- **Hyperspectral Image Analysis**: Process .tif, .npy, and other image formats
- **Sensor Data Integration**: Incorporate environmental sensor data
- **CNN-based Health Prediction**: Deep learning model for crop health assessment
- **Vegetation Indices**: Calculate NDVI, EVI, SAVI, NDWI, PRI, and more
- **Health Maps Generation**: Visual health maps with color-coded overlays
- **Comprehensive Reports**: HTML reports with analysis details
- **Risk Assessment**: Disease and pest risk evaluation
- **RESTful API**: Clean, well-documented API endpoints

## Project Structure

```
api1/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── exceptions.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── output_service.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── __init__.py
│   ├── config.py
│   └── main.py
├── data/
│   └── sample/
├── models/
├── outputs/
├── uploads/
├── convert_structured_data.py
├── test_api.py
├── requirements.txt
├── Procfile
└── README.md
```

## Installation

1. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create sample data and train model** (optional):
   ```bash
   python convert_structured_data.py --create-data --samples 500
   python convert_structured_data.py --train-model --epochs 20
   ```

## Usage

### Starting the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### API Endpoints

#### 1. Model API with JSON Input
```bash
POST /api/v1/predict/model-api
```

**Request Body**:
```json
{
  "field_id": "field_001",
  "user_id": "user_001",
  "hyperspectral_image_url": "http://example.com/image.tif",
  "sensor_data": {
    "air_temperature": 25.5,
    "humidity": 65.0,
    "soil_moisture": 45.0,
    "timestamp": "2025-10-04T10:00:00Z",
    "wind_speed": 3.2,
    "rainfall": 2.5,
    "solar_radiation": 800.0,
    "leaf_wetness": 35.0,
    "co2_level": 420.0,
    "ph_level": 6.8
  }
}
```

#### 2. File Upload API
```bash
POST /api/v1/predict/upload-files
```

**Form Data**:
- `field_id`: Field identifier
- `user_id`: User identifier
- `tif_file`: .tif image file (optional)
- `npy_file`: .npy data file (optional)
- `csv_file`: CSV sensor data file (optional)

#### 3. Health Check
```bash
GET /health
```

#### 4. API Information
```bash
GET /info
```

### Response Format

```json
{
  "status": "success",
  "message": "Analysis completed successfully",
  "health_map_url": "http://localhost:8000/static/outputs/health_map_field_001_20251004_120000.png",
  "indices_data_url": "http://localhost:8000/static/outputs/indices_data_field_001_20251004_120000.json",
  "report_url": "http://localhost:8000/static/outputs/analysis_report_field_001_20251004_120000.html",
  "sensor_data_url": "http://localhost:8000/static/outputs/indices_data_field_001_20251004_120000.csv",
  "health_indices": {
    "ndvi": 0.75,
    "evi": 0.68,
    "savi": 0.72,
    "ndwi": 0.45,
    "pri": 0.02,
    "chlorophyll_content": 38.5,
    "leaf_area_index": 4.2,
    "water_stress_index": 25.0
  },
  "crop_health_status": {
    "overall_health_score": 78.5,
    "health_category": "Good",
    "stress_indicators": [],
    "recommendations": [
      "Maintain current management practices",
      "Continue regular monitoring",
      "Prepare for harvest optimization"
    ],
    "disease_risk": 15.2,
    "pest_risk": 12.8
  },
  "metadata": {
    "field_id": "field_001",
    "processing_time": "2025-10-04T12:00:00Z",
    "model_version": "1.0.0",
    "processing_duration": 2.45,
    "image_resolution": "224x224",
    "bands_analyzed": 3
  }
}
```

## Testing

Run the comprehensive test suite:

```bash
python test_api.py --url http://localhost:8000
```

Test specific endpoints:
```bash
python test_api.py --test health
python test_api.py --test model-api
python test_api.py --test upload
```

## Configuration

Environment variables can be set to configure the application:

- `DEBUG`: Enable debug mode (default: false)
- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to the ML model (default: models/spectral_cnn.h5)
- `MAX_FILE_SIZE`: Maximum file size in bytes (default: 50MB)
- `LOG_LEVEL`: Logging level (default: INFO)

## Deployment

### Local Development
```bash
uvicorn app.main:app --reload
```

### Production (Gunicorn)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Heroku
1. Ensure `Procfile` exists with: `web: uvicorn app.main:app --host 0.0.0.0 --port $PORT`
2. Deploy using Heroku CLI or GitHub integration

## Health Categories

The API classifies crop health into five categories:

1. **Excellent** (80-100%): Optimal health, minimal intervention needed
2. **Good** (60-79%): Healthy crops, maintain current practices
3. **Fair** (40-59%): Moderate health, monitor closely
4. **Poor** (20-39%): Poor health, intervention recommended
5. **Critical** (0-19%): Severe issues, immediate action required

## Vegetation Indices

- **NDVI**: Normalized Difference Vegetation Index
- **EVI**: Enhanced Vegetation Index
- **SAVI**: Soil Adjusted Vegetation Index
- **NDWI**: Normalized Difference Water Index
- **PRI**: Photochemical Reflectance Index
- **Chlorophyll Content**: Estimated chlorophyll content (mg/g)
- **LAI**: Leaf Area Index
- **Water Stress Index**: Water stress percentage

## Support

For questions, issues, or contributions, please refer to the API documentation or contact the development team.

## License

This project is licensed under the MIT License.