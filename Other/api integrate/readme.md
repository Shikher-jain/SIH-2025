# CNN Multi-Modal Prediction API

A production-ready FastAPI application for multi-modal machine learning predictions using TIF images, NPY arrays, and CSV sensor data.

## 🚀 Features

- **Multi-Modal Input Support**: Process TIF images, NPY arrays, and CSV files
- **RESTful API**: Clean FastAPI endpoints with automatic documentation
- **Production Ready**: Docker support, logging, error handling
- **Flexible Configuration**: Environment-based configuration
- **Health Checks**: Built-in health and status endpoints

## 📋 Prerequisites

- Python 3.8+
- TensorFlow 2.x
- Your trained CNN model file (model.h5)

## 🛠️ Installation

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd api-integrate
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Add your trained model**
   - Place your trained model file as `model.h5` in the project root
   - Or update `MODEL_PATH` in your `.env` file

6. **Run the application**
   ```bash
   # Development mode
   python -m uvicorn app.main:app --reload
   
   # Or use the deploy script
   python deploy.py
   ```

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t cnn-api .
   ```

2. **Run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

## 📡 API Endpoints

### Base URL
```
http://localhost:8000
```

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| GET | `/docs` | Interactive API documentation (Swagger) |
| GET | `/redoc` | Alternative API documentation |
| GET | `/predict/health` | Health check endpoint |
| GET | `/predict/status` | API status and model information |
| POST | `/predict/` | Make predictions with JSON data |
| POST | `/predict/files` | Make predictions with file uploads |

## 📤 Usage Examples

### 1. JSON Prediction

```bash
curl -X POST "http://localhost:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{
       "image_data": [[1, 2, 3], [4, 5, 6]],
       "npy_data": [0.1, 0.2, 0.3],
       "csv_data": [[1.0, 2.0], [3.0, 4.0]]
     }'
```

### 2. File Upload Prediction

```bash
curl -X POST "http://localhost:8000/predict/files" \
     -F "tif_file=@sample.tif" \
     -F "npy_file=@data.npy" \
     -F "csv_file=@sensor_data.csv"
```

### 3. Health Check

```bash
curl "http://localhost:8000/predict/health"
```

## 📁 Project Structure

```
api-integrate/
├── app/
│   ├── core/
│   │   ├── __init__.py
│   │   └── exceptions.py          # Exception handlers
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py          # CNN model wrapper
│   ├── routers/
│   │   ├── __init__.py
│   │   └── predict.py            # Prediction endpoints
│   ├── schemas/
│   │   ├── __init__.py
│   │   └── prediction.py         # Pydantic models
│   ├── utils/
│   │   ├── __init__.py
│   │   └── preprocessing.py      # Data preprocessing
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   └── main.py                   # FastAPI application
├── tests/                        # Test suite
├── sample_data/                  # Sample test files
├── .env.example                  # Environment template
├── .gitignore                    # Git ignore rules
├── deploy.py                     # Deployment script
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # Docker image definition
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```env
# Model Configuration
MODEL_PATH=model.h5
IMAGE_INPUT_WIDTH=224
IMAGE_INPUT_HEIGHT=224
IMAGE_INPUT_CHANNELS=3

# API Configuration
DEBUG=false
HOST=0.0.0.0
PORT=8000

# File Upload Limits
MAX_FILE_SIZE=52428800  # 50MB
```

## 📊 Input Data Formats

### TIF Images
- **Format**: TIFF image files
- **Supported**: .tif, .tiff
- **Processing**: Resized to model input dimensions

### NPY Arrays
- **Format**: NumPy binary files
- **Supported**: .npy
- **Content**: Preprocessed numerical data

### CSV Sensor Data
- **Format**: Comma-separated values
- **Supported**: .csv
- **Content**: Sensor readings with headers

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest tests/ --cov=app

# Run specific test
pytest tests/test_predict.py::test_prediction_endpoint
```

## 📝 Development

### Adding New Features

1. **Models**: Add new model classes in `app/models/`
2. **Endpoints**: Add new routes in `app/routers/`
3. **Schemas**: Define request/response models in `app/schemas/`
4. **Utils**: Add helper functions in `app/utils/`

### Code Quality

```bash
# Format code
black app/

# Lint code
flake8 app/

# Type checking
mypy app/
```

## 🚀 Production Deployment

### Docker Production

```bash
# Build production image
docker build -t cnn-api:latest .

# Run with production settings
docker run -d \
  --name cnn-api \
  -p 8000:8000 \
  -e DEBUG=false \
  -e WORKERS=4 \
  -v /path/to/model.h5:/app/model.h5 \
  cnn-api:latest
```

### Performance Tuning

- **Workers**: Adjust `WORKERS` environment variable
- **Memory**: Monitor model memory usage
- **Caching**: Implement Redis for prediction caching

## 🔍 Monitoring

### Health Checks
- `/predict/health` - Basic health status
- `/predict/status` - Detailed API and model status

### Logging
- Structured JSON logging
- Configurable log levels
- Request/response logging

## 🛠️ Troubleshooting

### Common Issues

1. **Model not found**
   - Ensure `model.h5` exists in the specified path
   - Check `MODEL_PATH` in `.env` file

2. **Memory errors**
   - Reduce batch size
   - Increase available memory
   - Optimize model architecture

3. **File upload errors**
   - Check `MAX_FILE_SIZE` setting
   - Verify file format compatibility

## 📄 License

[Your License Here]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Support

For issues and questions, please open an issue on GitHub or contact [your-email@example.com]