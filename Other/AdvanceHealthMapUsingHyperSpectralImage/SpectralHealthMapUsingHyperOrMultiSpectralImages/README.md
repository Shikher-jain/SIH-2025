# AI-Powered Spectral Health Mapping System API

A modern FastAPI-based API for spectral image analysis and agricultural health monitoring using advanced AI models.

## 🚀 Features

- **Multi-Model AI Integration**: BERT-based text analysis, CNN disease detection, LSTM temporal analysis, U-Net segmentation
- **Spectral Data Analysis**: Hyperspectral and multispectral image processing
- **Disease Detection**: Automated plant disease identification using deep learning
- **Anomaly Detection**: Advanced anomaly detection using autoencoders
- **Risk Assessment**: Comprehensive field health risk evaluation
- **User Management**: JWT-based authentication with role-based access control
- **RESTful API**: Clean, documented API endpoints with automatic OpenAPI documentation
- **Database Integration**: SQLAlchemy with PostgreSQL/SQLite support
- **File Upload**: Support for various spectral data formats (.tif, .npy, .mat, .h5, etc.)

## 📁 Project Structure

```
project_name/
│
├── app/
│   ├── main.py              # Entry point
│   ├── core/                # Core settings & config
│   │   ├── config.py        # Environment variables, constants
│   │   └── security.py      # Auth/JWT logic
│   │
│   ├── api/                 # API layer (routes)
│   │   ├── v1/              # Versioning your APIs
│   │   │   ├── routes/
│   │   │   │   ├── users.py
│   │   │   │   ├── auth.py
│   │   │   │   ├── spectral_analysis.py
│   │   │   │   └── models.py
│   │   │   └── __init__.py
│   │   └── deps.py          # Dependencies (e.g., get_db, oauth2_scheme)
│   │
│   ├── models/              # SQLAlchemy models
│   │   ├── user.py
│   │   ├── spectral.py
│   │   └── __init__.py
│   │
│   ├── schemas/             # Pydantic schemas (request/response validation)
│   │   ├── user.py
│   │   ├── spectral.py
│   │   └── common.py
│   │
│   ├── services/            # Business logic (keep routes thin)
│   │   ├── user_service.py
│   │   ├── spectral_service.py
│   │   └── model_integration.py
│   │
│   ├── db/                  # Database stuff
│   │   ├── session.py       # DB session creation
│   │   ├── base.py          # Base metadata
│   │   └── init_db.py
│   │
│   ├── tests/               # Unit/integration tests
│   │   ├── test_users.py
│   │   └── test_spectral.py
│   │
│   └── utils/               # Helpers (emails, file handling, etc.)
│       ├── logger.py
│       ├── email.py
│       └── file_handler.py
│
├── src/                     # Original spectral processing modules
│   ├── analytics/
│   ├── data/
│   ├── dashboard/
│   └── models/
│
├── alembic/                 # DB migrations (if SQLAlchemy)
├── models/saved/            # Trained model files
├── uploads/                 # File upload directory
├── logs/                    # Application logs
│
├── requirements.txt         # Dependencies
├── .env                     # Environment variables
├── alembic.ini              # Alembic config
└── README.md
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd SpectralHealthMapUsingHyperOrMultiSpectralImages
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
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

5. **Initialize database**
   ```bash
   python -c "from app.db.init_db import init_db, create_initial_data; init_db(); create_initial_data()"
   ```

## 🚀 Running the Application

### Development Server
```bash
# Run with uvicorn directly
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Or using the main module
python -m app.main
```

### Production Server
```bash
# Install gunicorn for production
pip install gunicorn

# Run with gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 API Documentation

Once the server is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **ReDoc documentation**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔐 Authentication

The API uses JWT-based authentication. To access protected endpoints:

1. **Register a new user**:
   ```bash
   POST /api/v1/auth/register
   ```

2. **Login to get access token**:
   ```bash
   POST /api/v1/auth/login
   ```

3. **Use token in requests**:
   ```bash
   Authorization: Bearer <your-access-token>
   ```

## 🧪 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - Register new user
- `POST /api/v1/auth/login` - Login user
- `POST /api/v1/auth/refresh` - Refresh access token

### User Management
- `GET /api/v1/users/me` - Get current user info
- `PUT /api/v1/users/me` - Update current user
- `GET /api/v1/users/` - List all users (admin only)

### Spectral Analysis
- `POST /api/v1/spectral/analyze` - Analyze spectral data
- `POST /api/v1/spectral/upload-analyze` - Upload and analyze file
- `GET /api/v1/spectral/fields` - List analyzed fields
- `GET /api/v1/spectral/fields/{field_id}` - Get field analysis
- `POST /api/v1/spectral/batch-analyze` - Batch analysis

### Model Integration
- `POST /api/v1/models/predict/disaster-text` - Predict disaster from text
- `POST /api/v1/models/predict/spectral-disease` - Disease detection
- `POST /api/v1/models/predict/anomaly-detection` - Anomaly detection
- `POST /api/v1/models/predict/segmentation` - Image segmentation
- `POST /api/v1/models/predict/comprehensive` - Multi-model analysis
- `GET /api/v1/models/status` - Get model status
- `GET /api/v1/models/health` - Health check models

## 🧠 AI Models

The system integrates multiple AI models:

1. **BERT Disaster Classification**: Text-based disaster type and urgency prediction
2. **Spectral CNN**: Disease detection from hyperspectral data
3. **Spectral LSTM**: Temporal analysis and progression prediction
4. **Autoencoder**: Anomaly detection in spectral signatures
5. **U-Net Segmentation**: Pixel-wise classification of vegetation health
6. **Random Forest**: Fallback classification for robustness

## 📊 Data Formats

Supported file formats:
- **Hyperspectral**: .tif, .tiff, .h5, .hdf5
- **Multispectral**: .tif, .tiff
- **Numpy arrays**: .npy, .npz
- **MATLAB files**: .mat

## 🔧 Configuration

Key configuration options in `.env`:

```env
# Server settings
DEBUG=true
HOST=127.0.0.1
PORT=8000

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/spectral_health_db

# Models
MODEL_PATH=models/saved
ENABLE_GPU=true
HYPERSPECTRAL_BANDS=224

# Analysis thresholds
ANOMALY_THRESHOLD=0.7
DISEASE_THRESHOLD=0.8
STRESS_THRESHOLD=0.6
```

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest app/tests/test_users.py
```

## 📦 Database Migrations

Using Alembic for database migrations:

```bash
# Generate migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Downgrade
alembic downgrade -1
```

## 🚨 Monitoring and Logging

- **Logging**: Configured with rotation, multiple levels
- **Health checks**: Built-in model and system health endpoints
- **Error tracking**: Comprehensive error logging and reporting

## 🔒 Security Features

- JWT-based authentication
- Role-based access control (user, researcher, admin)
- Input validation with Pydantic
- File upload security
- CORS configuration
- Password hashing with bcrypt

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables for Production
- Set `DEBUG=false`
- Configure proper database URL
- Set strong `SECRET_KEY`
- Configure email settings for notifications
- Set up Redis for caching (optional)

## 📈 Performance

- **Async/await**: Full async support for concurrent requests
- **Database connection pooling**: Efficient database connections
- **Model caching**: Models loaded once at startup
- **File streaming**: Efficient file upload handling
- **Background tasks**: Async processing for heavy operations

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the API documentation at `/docs`

## 🔄 Version History

- **v1.0.0**: Initial FastAPI implementation with model integration
- **v0.x**: Original dashboard-based system

---

**Built with ❤️ using FastAPI, SQLAlchemy, and modern AI/ML technologies**