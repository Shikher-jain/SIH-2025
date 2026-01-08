# Multi-Modal CNN API Deployment Guide

## Quick Start

### 1. Install Dependencies
```bash
cd "c:\shikher_jain\SIH\api integrate"
pip install -r requirements.txt
```

### 2. Deploy the API

#### Development Mode:
```bash
python -m app.main
```

#### Production Mode:
```bash
python deploy.py
```

#### Custom Configuration:
```bash
python deploy.py --host 0.0.0.0 --port 8080 --workers 2
```

### 3. Access the API

- **API Documentation**: http://localhost:8000/docs
- **API Status**: http://localhost:8000/predict/status
- **Health Check**: http://localhost:8000/predict/health
- **Main Endpoint**: http://localhost:8000/

## API Endpoints

### 1. JSON-based Prediction (Recommended for JS backends)
**POST** `/predict/`
```json
{
  "fieldId": "field_001",
  "userId": "user_123", 
  "hyperSpectralImageUrl": "https://example.com/image.npy",
  "sensorData": {
    "airTemperature": 25.5,
    "humidity": 60.0,
    "soilMoisture": 30.0,
    "timestamp": "2025-10-04T10:00:00Z",
    "windSpeed": 10.0,
    "rainfall": 0.0,
    "solarRadiation": 800.0,
    "leafWetness": 15.0,
    "co2Level": 400.0,
    "phLevel": 6.5
  }
}
```

### 2. File Upload Prediction
**POST** `/predict/files`
- Upload: `tif_file`, `npy_file`, `csv_file`

### 3. Status and Health
- **GET** `/predict/status` - API status
- **GET** `/predict/health` - Model health
- **GET** `/` - API information

## JavaScript Integration

### Using the provided client:
```javascript
const { MultiModalCNNClient } = require('./js_client_example.js');

const client = new MultiModalCNNClient('http://your-api-server:8000');

// Check status
const status = await client.getStatus();

// Make prediction
const result = await client.predictWithUrls({
  fieldId: "field_001",
  userId: "user_123",
  hyperSpectralImageUrl: "https://example.com/data.npy",
  sensorData: { /* sensor data */ }
});
```

### Direct fetch example:
```javascript
// Check API status
const response = await fetch('http://localhost:8000/predict/status');
const status = await response.json();

// Make prediction
const prediction = await fetch('http://localhost:8000/predict/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    fieldId: "field_001",
    userId: "user_123", 
    hyperSpectralImageUrl: "https://example.com/image.npy",
    sensorData: {
      airTemperature: 25.5,
      humidity: 60.0,
      // ... other sensor data
    }
  })
});
```

## Production Considerations

### 1. CORS Configuration
Update `app/main.py` to restrict origins:
```python
allow_origins=["https://yourdomain.com", "https://app.yourdomain.com"]
```

### 2. Environment Variables
Set these for production:
```bash
DEBUG=False
MODEL_PATH=/path/to/your/model.h5
```

### 3. SSL/HTTPS
Use a reverse proxy like nginx for SSL termination.

### 4. Monitoring
Monitor the `/predict/health` endpoint for service availability.

## Response Format

Successful prediction:
```json
{
  "prediction": [[0.1, 0.2, 0.7, 0.0]],
  "status": "success",
  "message": "Prediction completed successfully",
  "fieldId": "field_001",
  "processingTime": 2.34
}
```

Error response:
```json
{
  "status": "error",
  "message": "Error description",
  "detail": "Detailed error information"
}
```

## Testing

Test the API using curl:
```bash
# Status check
curl http://localhost:8000/predict/status

# Health check  
curl http://localhost:8000/predict/health

# Prediction (requires valid data)
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "fieldId": "test_field",
    "userId": "test_user",
    "hyperSpectralImageUrl": "https://example.com/test.npy",
    "sensorData": {
      "airTemperature": 25.0,
      "humidity": 60.0,
      "soilMoisture": 30.0,
      "timestamp": "2025-10-04T10:00:00Z",
      "windSpeed": 10.0,
      "rainfall": 0.0,
      "solarRadiation": 800.0,
      "leafWetness": 15.0,
      "co2Level": 400.0,
      "phLevel": 6.5
    }
  }'
```