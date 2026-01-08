# Earth Engine Agricultural Data Processing Pipeline - Complete Tutorial

## Overview

This tutorial covers the complete workflow for processing agricultural data using Google Earth Engine, including satellite imagery, sensor data, and weather information. The pipeline processes geographical areas cell by cell, collecting all necessary data types and creating comprehensive metadata.

## Architecture Overview

The system is built with modular components:

- **Main Pipeline (`main.js`)**: Orchestrates the entire process
- **Earth Engine Manager**: Handles Google Earth Engine initialization and operations
- **Data Downloader**: Manages satellite, sensor, and weather data downloads
- **Weather Downloader**: Specialized weather data collection from OpenWeather API
- **Metadata Manager**: Creates comprehensive metadata files
- **Grid Processor**: Handles geographical grid generation
- **Configuration Manager**: Manages all settings

## Core Components and Methods

### 1. Earth Engine Initialization

#### Key Method: `initialize()`

```javascript
async initialize() {
  // Authenticate with service account
  const privateKey = JSON.parse(fs.readFileSync('earth-engine-service-account.json', 'utf8'));
  
  await new Promise((resolve, reject) => {
    ee.data.authenticateViaPrivateKey(privateKey, (error) => {
      if (error) reject(new Error(`Authentication failed: ${error}`));
      else resolve();
    });
  });
  
  // Initialize Earth Engine
  await new Promise((resolve, reject) => {
    ee.initialize(null, null, (error) => {
      if (error) reject(new Error(`Initialization failed: ${error}`));
      else resolve();
    });
  });
}
```

**Purpose**: Establishes connection to Google Earth Engine using service account credentials.

### 2. Satellite Data Collection

#### Key Method: `downloadSatelliteData(cell, outputDir)`

```javascript
async downloadSatelliteData(cell, outputDir) {
  // Create geometry from cell polygon
  const region = this.ee.createGeometry(cell.polygon);
  
  // Calculate date range (last 5 months by default)
  const dateRange = this.calculateDateRange(satelliteConfig.dateRangeMonths);
  
  // Get Sentinel-2 collection with cloud filtering
  let collection = this.ee.getSentinel2Collection(
    region,
    dateRange.start,
    dateRange.end,
    satelliteConfig.cloudThreshold
  );
  
  // Get most recent image
  const mostRecentImage = collection.first().clip(region);
  
  // Select and rename specific bands
  const multispecImage = mostRecentImage
    .select(['B2', 'B3', 'B4', 'B5', 'B8', 'B11'])
    .rename(['Blue_B2', 'Green_B3', 'Red_B4', 'RedEdge_B5', 'NIR_B8', 'SWIR_B11']);
  
  // Generate download URL
  const downloadUrl = await this.ee.getDownloadUrl(multispecImage, {
    scale: 10,
    region: region,
    format: 'GEO_TIFF',
    maxPixels: 1e10
  });
  
  // Download file
  await this.downloadFile(downloadUrl, filepath, 'satellite');
  
  return {
    success: true,
    filepath: filepath,
    filename: filename,
    fileSize: fileSizeMB,
    imageDate: imageDate,
    bands: this.createBandMetadata(satelliteConfig)
  };
}
```

**Purpose**: Downloads Sentinel-2 satellite imagery with specific spectral bands for vegetation and agricultural analysis.

**Key Features**:
- Cloud filtering (< 50% cloud cover)
- Multi-spectral band selection (Blue, Green, Red, Red Edge, NIR, SWIR)
- Automatic fallback for extended date ranges
- 10m spatial resolution

### 3. Sensor Data Collection

#### Key Method: `downloadSensorData(cell, outputDir)`

```javascript
async downloadSensorData(cell, outputDir) {
  const region = this.ee.createGeometry(cell.polygon);
  const sensorConfig = this.config.get('sensor');
  
  // Load multiple sensor datasets
  const validSensors = [];
  for (const [sensorName, assetId] of Object.entries(sensorConfig.assets)) {
    try {
      const sensorImage = this.ee.getSensorImage(assetId);
      const imageInfo = await this.ee.getImageInfo(sensorImage);
      
      // Create proper band names
      const bandNames = imageInfo.bands.map((band, index) => 
        `${sensorName}_${band.id || `Layer${index + 1}`}`
      );
      
      const renamedImage = sensorImage.rename(bandNames);
      validSensors.push({
        name: sensorName,
        image: renamedImage,
        bandCount: imageInfo.bands.length,
        bandNames: bandNames
      });
    } catch (error) {
      console.warn(`Failed to load ${sensorName}: ${error.message}`);
    }
  }
  
  // Combine all sensor images
  const ee = require('@google/earthengine');
  const images = validSensors.map(sensor => sensor.image);
  const combinedSensorImage = ee.Image.cat(images);
  
  // Download combined sensor data
  const downloadUrl = await this.ee.getDownloadUrl(combinedSensorImage, {
    scale: 1000,
    region: region,
    format: 'GEO_TIFF',
    maxPixels: 1e10
  });
  
  await this.downloadFile(downloadUrl, filepath, 'sensor');
  
  return {
    success: true,
    filepath: filepath,
    sensorsLoaded: validSensors.map(s => s.name),
    totalBands: totalBands,
    bands: this.createSensorBandMetadata(validSensors)
  };
}
```

**Purpose**: Downloads soil property data from multiple sensor sources and combines them into a single multi-band image.

**Sensor Types**:
- **ECe**: Electrical Conductivity (soil salinity)
- **N**: Nitrogen content (soil fertility)
- **P**: Phosphorus content (soil fertility)
- **OC**: Organic Carbon content (soil health)
- **pH**: Soil acidity/alkalinity

### 4. Weather Data Collection

#### Key Method: `downloadWeatherData(cell, outputDir)`

```javascript
async downloadWeatherData(cell, outputDir) {
  // Get cell center coordinates
  const [lon, lat] = cell.getCenter();
  
  // Get current weather
  const currentWeather = await this.getCurrentWeather(lat, lon);
  
  // Get forecast data (processed into daily summaries)
  const forecastWeather = await this.getDailyForecastFromFree(lat, lon);
  
  // Process and enrich weather data
  const processedData = this.processWeatherData(currentWeather, forecastWeather, cell);
  
  // Save to JSON file
  fs.writeFileSync(filepath, JSON.stringify(processedData, null, 2), 'utf8');
  
  return {
    success: true,
    filepath: filepath,
    fileSize: fileSizeMB,
    dataPoints: this.countDataPoints(currentWeather, forecastWeather),
    fields: this.weatherConfig.fields
  };
}
```

#### Sub-method: `getCurrentWeather(lat, lon)`

```javascript
async getCurrentWeather(lat, lon) {
  const url = `${this.weatherConfig.baseUrl}/weather?lat=${lat}&lon=${lon}&appid=${this.weatherConfig.apiKey}&units=metric`;
  
  const data = await this.makeHttpRequest(url);
  return JSON.parse(data);
}
```

#### Sub-method: `getDailyForecastFromFree(lat, lon)`

```javascript
async getDailyForecastFromFree(lat, lon) {
  // Get 5-day/3-hour forecast (free API)
  const forecastData = await this.getForecastWeather(lat, lon);
  
  // Process 3-hour data into daily summaries
  const dailyData = this.processToDailyForecast(forecastData, days);
  
  return dailyData;
}
```

**Purpose**: Collects current weather and forecast data from OpenWeather API, processing 3-hour intervals into daily summaries.

**Weather Fields**:
- Temperature (current, min, max, feels-like)
- Humidity (percentage)
- Atmospheric pressure (hPa)
- Wind speed and direction
- Precipitation (rain/snow)
- Cloud coverage

### 5. Metadata Creation

#### Key Method: `createCellMetadata(cell, satelliteResult, sensorResult, outputDir, weatherResult)`

```javascript
createCellMetadata(cell, satelliteResult, sensorResult, outputDir, weatherResult = null) {
  const bounds = cell.getBounds();
  const metadata = {
    // Cell identification
    cell_id: cell.id,
    area_name: this.config.get('area.name'),
    generation_date: new Date().toISOString(),
    
    // Geographical information
    geometry: {
      bounds: bounds,
      center: cell.getCenter(),
      area_km2: cell.areaKm2,
      polygon: cell.polygon
    },
    
    // Processing information
    processing_status: {
      completed: cell.processed,
      retry_count: cell.retryCount,
      processing_time_ms: cell.processingTime,
      error: cell.error
    },
    
    // Data metadata
    satellite_data: satelliteResult ? this.createSatelliteMetadata(satelliteResult) : null,
    sensor_data: sensorResult ? this.createSensorMetadata(sensorResult) : null,
    weather_data: weatherResult ? this.createWeatherMetadata(weatherResult) : null,
    
    // Configuration used
    configuration: {
      grid_config: this.config.get('grid'),
      satellite_config: this.config.get('satellite'),
      sensor_config: this.config.get('sensor'),
      weather_config: this.config.get('weather')
    },
    
    // Quality metrics
    quality_metrics: this.calculateQualityMetrics(satelliteResult, sensorResult, weatherResult)
  };
  
  // Save metadata file
  const metadataPath = path.join(outputDir, `cell_${cell.id}_metadata.json`);
  fs.writeFileSync(metadataPath, JSON.stringify(metadata, null, 2));
  
  return { success: true, filepath: metadataPath, metadata: metadata };
}
```

**Purpose**: Creates comprehensive metadata describing all collected data, processing parameters, and quality metrics.

## 6. Complete Processing Function with Promise.all

#### Method: `processAllDataConcurrently(cell, outputDir)`

Here's how to implement a function that gets all data types simultaneously using Promise.all:

```javascript
async processAllDataConcurrently(cell, outputDir) {
  console.log(`🔄 Starting concurrent data collection for cell ${cell.id}...`);
  
  try {
    // Create cell output directory
    cell.outputDir = this.fileUtils.createCellOutputDir(cell.id, outputDir);
    
    // Start all downloads concurrently
    const [satelliteResult, sensorResult, weatherResult] = await Promise.all([
      // Satellite data download
      this.dataDownloader.downloadSatelliteData(cell, cell.outputDir)
        .then(result => {
          console.log(`   ✅ Satellite completed: ${result.fileSize.toFixed(2)} MB`);
          return result;
        })
        .catch(error => {
          console.log(`   ❌ Satellite failed: ${error.message}`);
          return null;
        }),
      
      // Sensor data download
      this.dataDownloader.downloadSensorData(cell, cell.outputDir)
        .then(result => {
          console.log(`   ✅ Sensor completed: ${result.fileSize.toFixed(2)} MB`);
          return result;
        })
        .catch(error => {
          console.log(`   ❌ Sensor failed: ${error.message}`);
          return null;
        }),
      
      // Weather data download
      this.dataDownloader.downloadWeatherData(cell, cell.outputDir)
        .then(result => {
          console.log(`   ✅ Weather completed: ${result.fileSize.toFixed(2)} MB`);
          return result;
        })
        .catch(error => {
          console.log(`   ❌ Weather failed: ${error.message}`);
          return null;
        })
    ]);
    
    console.log(`   📊 Data collection summary:`);
    console.log(`   - Satellite: ${satelliteResult ? '✅' : '❌'}`);
    console.log(`   - Sensor: ${sensorResult ? '✅' : '❌'}`);
    console.log(`   - Weather: ${weatherResult ? '✅' : '❌'}`);
    
    // Create comprehensive metadata
    const metadataResult = this.metadataManager.createCellMetadata(
      cell, satelliteResult, sensorResult, cell.outputDir, weatherResult
    );
    
    console.log(`   📋 Metadata created: ${metadataResult.filepath}`);
    
    return {
      success: true,
      satellite: satelliteResult,
      sensor: sensorResult,
      weather: weatherResult,
      metadata: metadataResult
    };
    
  } catch (error) {
    console.error(`   ❌ Concurrent processing failed: ${error.message}`);
    throw error;
  }
}
```

## Complete Workflow Example

Here's how a complete processing cycle works for one cell:

```javascript
async function processSingleCell(cell) {
  const startTime = Date.now();
  
  console.log(`\n🔲 Processing Cell ${cell.id}`);
  console.log(`   📍 Center: ${cell.getCenter().join(', ')}`);
  console.log(`   📐 Area: ${cell.areaKm2.toFixed(1)} km²`);
  
  try {
    // 1. Initialize Earth Engine connection
    await earthEngine.refreshConnection();
    
    // 2. Create output directory
    const outputDir = fileUtils.createCellOutputDir(cell.id, baseOutputDir);
    
    // 3. Get all data types concurrently
    const results = await processAllDataConcurrently(cell, outputDir);
    
    // 4. Calculate processing time
    const processingTime = Date.now() - startTime;
    console.log(`   ⏱️ Processing completed in ${(processingTime / 1000).toFixed(1)}s`);
    
    // 5. Update cell status
    cell.markSuccess(results.satellite, results.sensor, processingTime);
    
    return results;
    
  } catch (error) {
    console.error(`   ❌ Cell processing failed: ${error.message}`);
    cell.markError(error.message);
    throw error;
  }
}
```

## Configuration

The system uses a comprehensive configuration file (`config.json`):

```json
{
  "area": {
    "name": "1mil_weatherTest-1",
    "polygon": [[76.663557, 30.526306], [77.630597, 30.526306], [77.630597, 29.661325], [76.663557, 29.661325], [76.663557, 30.526306]]
  },
  "satellite": {
    "cloudThreshold": 50,
    "dateRangeMonths": 5,
    "scale": 10,
    "bands": ["B2", "B3", "B4", "B5", "B8", "B11"],
    "bandNames": ["Blue_B2", "Green_B3", "Red_B4", "RedEdge_B5", "NIR_B8", "SWIR_B11"]
  },
  "sensor": {
    "scale": 1000,
    "assets": {
      "ECe": "projects/pk07007/assets/ECe",
      "N": "projects/pk07007/assets/N",
      "P": "projects/pk07007/assets/P",
      "OC": "projects/pk07007/assets/OC",
      "pH": "projects/pk07007/assets/pH"
    }
  },
  "weather": {
    "apiKey": "your-openweather-api-key",
    "forecastDays": 3,
    "fields": ["temperature", "humidity", "pressure", "windSpeed", "windDirection", "precipitation", "cloudCover"]
  }
}
```

## Output Structure

For each processed cell, the system creates:

```
cell_1_1/
├── cell_1_1_metadata.json                    # Comprehensive metadata
├── 1mil_weatherTest-1_satellite_2025-09-20_timestamp.tif  # Satellite imagery
├── 1mil_weatherTest-1_sensor_timestamp.tif   # Sensor data
└── 1mil_weatherTest-1_weather_timestamp.json # Weather data
```

## Key Features

1. **Concurrent Data Collection**: All data types are downloaded simultaneously using Promise.all for maximum efficiency

2. **Robust Error Handling**: Each data source has individual error handling without affecting others

3. **Comprehensive Metadata**: Detailed metadata includes band information, quality metrics, and usage recommendations

4. **Quality Assessment**: Automatic quality scoring based on data completeness and characteristics

5. **Flexible Configuration**: All parameters can be adjusted through the configuration file

6. **Progress Tracking**: Real-time progress updates and processing statistics

## Usage Instructions

1. **Setup**: Ensure Google Earth Engine service account credentials and OpenWeather API key are configured

2. **Configure**: Modify `config.json` with your area of interest and processing parameters

3. **Run**: Execute `node main.js` to start the processing pipeline

4. **Monitor**: Follow console output for real-time progress updates

5. **Results**: Find processed data and metadata in the output directory structure

This tutorial covers the complete data processing workflow, focusing on the core functions for satellite imagery, sensor data, weather information collection, and metadata generation as implemented in the actual codebase.