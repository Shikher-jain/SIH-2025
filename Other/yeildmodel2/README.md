# NDVI and Yield API

This project provides an API for processing NDVI (Normalized Difference Vegetation Index) and yield data, generating visual representations of crop health and yield values. The API returns a red-yellow-green mask based on yield values, allowing for threshold adjustments.

## Project Structure

```
ndvi-yield-api
├── src
│   ├── app.py                # Entry point for the API application
│   ├── ndvi_processor.py     # Processes NDVI data from in-memory numpy arrays
│   ├── sensor_processor.py    # Processes sensor data and generates yield masks
│   ├── config.json           # Configuration settings for the application
│   ├── models
│   │   └── schemas.py        # Data models and schemas for input/output validation
│   └── utils
│       └── io.py             # Utility functions for handling numpy arrays and masks
├── tests
│   └── test_api.py           # Unit tests for the API endpoints
├── requirements.txt          # List of dependencies for the project
├── .env                      # Environment variables for the application
├── Dockerfile                # Instructions for building a Docker image
└── README.md                 # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ndvi-yield-api
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables in the `.env` file as needed.

## Usage

1. Start the API server:
   ```
   python src/app.py
   ```

2. Access the API endpoints:
   - **Process NDVI Data**: Send a POST request to `/ndvi` with the NDVI data in the request body.
   - **Process Yield Data**: Send a POST request to `/yield` with the yield data and threshold parameters.

## API Documentation

- **POST /ndvi**
  - Request Body: NDVI data as a numpy array.
  - Response: Returns a red-yellow-green heatmap based on NDVI values.

- **POST /yield**
  - Request Body: Yield data as a numpy array with optional threshold adjustments.
  - Response: Returns a red-yellow-green mask based on yield values.

## Testing

Run the unit tests to ensure the API functions correctly:
```
pytest tests/test_api.py
```

## License

This project is licensed under the MIT License. See the LICENSE file for more details.