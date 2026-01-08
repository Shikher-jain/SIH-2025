# Chapter 5: Risk Assessment and Prediction

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand the principles of agricultural risk assessment
- Explain how the system combines multiple factors to generate risk maps
- Describe predictive modeling techniques for crop health
- Implement risk calculation algorithms
- Evaluate the effectiveness of risk assessment approaches

## Key Concepts 

### Introduction to Agricultural Risk Assessment

Agricultural risk assessment involves identifying, analyzing, and evaluating potential threats to crop health and yield. In the context of the AI-Powered Spectral Health Mapping System, risk assessment combines:

1. **Environmental Factors**: Weather conditions, soil properties, and climate patterns
2. **Spectral Analysis**: Vegetation indices, anomaly detection, and health indicators
3. **Temporal Patterns**: Historical data, progression rates, and seasonal trends
4. **Biological Factors**: Pest pressure, disease history, and plant stress indicators

### Risk Assessment Framework

The system uses a multi-dimensional risk assessment framework:

```
Comprehensive Risk Assessment
├── Environmental Risk Factors
│   ├── Temperature Extremes
│   ├── Humidity Levels
│   ├── Precipitation Patterns
│   ├── Wind Conditions
│   └── Soil Conditions
├── Spectral Risk Factors
│   ├── Vegetation Index Decline
│   ├── Anomaly Detection
│   ├── Stress Indicators
│   └── Health Classification
├── Temporal Risk Factors
│   ├── Progression Rates
│   ├── Seasonal Patterns
│   └── Historical Trends
└── Biological Risk Factors
    ├── Disease History
    ├── Pest Pressure
    └── Plant Stress Levels
```

## Environmental Risk Assessment

### Temperature Risk

Temperature extremes can significantly impact crop health:

```python
def calculate_temperature_risk(air_temperature):
    """
    Calculate risk based on air temperature
    """
    risk_score = 0.0
    
    # Heat stress
    if air_temperature > 35:  # °C
        risk_score += 0.4
    elif air_temperature > 30:
        risk_score += 0.2
    
    # Cold stress
    if air_temperature < 5:  # °C
        risk_score += 0.4
    elif air_temperature < 10:
        risk_score += 0.2
    
    return min(risk_score, 1.0)
```

### Humidity Risk

High humidity creates favorable conditions for fungal diseases:

```python
def calculate_humidity_risk(humidity):
    """
    Calculate risk based on humidity levels
    """
    risk_score = 0.0
    
    if humidity > 85:  # %
        risk_score += 0.3
    elif humidity > 75:
        risk_score += 0.15
    elif humidity < 30:
        risk_score += 0.1  # Low humidity stress
    
    return min(risk_score, 1.0)
```

### Precipitation Risk

Rainfall patterns affect both water stress and disease spread:

```python
def calculate_precipitation_risk(rainfall, leaf_wetness):
    """
    Calculate risk based on precipitation and leaf wetness
    """
    risk_score = 0.0
    
    # Heavy rainfall (disease spread risk)
    if rainfall > 15:  # mm in last 24 hours
        risk_score += 0.3
    
    # Moderate rainfall
    elif rainfall > 5:
        risk_score += 0.15
    
    # Leaf wetness duration (fungal disease risk)
    if leaf_wetness > 8:  # hours
        risk_score += 0.25
    elif leaf_wetness > 4:
        risk_score += 0.1
    
    return min(risk_score, 1.0)
```

### Integrated Environmental Risk

Combining multiple environmental factors:

```python
class EnvironmentalRiskEngine:
    def calculate_environmental_risk(self, sensor_data):
        """
        Calculate comprehensive environmental risk score
        """
        # Individual risk factors
        temp_risk = self.calculate_temperature_risk(
            sensor_data.get('air_temperature', 20)
        )
        
        humidity_risk = self.calculate_humidity_risk(
            sensor_data.get('humidity', 50)
        )
        
        precip_risk = self.calculate_precipitation_risk(
            sensor_data.get('rainfall', 0),
            sensor_data.get('leaf_wetness', 0)
        )
        
        wind_risk = self.calculate_wind_risk(
            sensor_data.get('wind_speed', 2)
        )
        
        soil_risk = self.calculate_soil_risk(
            sensor_data.get('soil_moisture', 40),
            sensor_data.get('soil_temperature', 18)
        )
        
        # Weighted combination
        # Weights can be adjusted based on crop type and region
        environmental_risk = (
            temp_risk * 0.25 +
            humidity_risk * 0.20 +
            precip_risk * 0.20 +
            wind_risk * 0.15 +
            soil_risk * 0.20
        )
        
        return min(environmental_risk, 1.0)
    
    def calculate_wind_risk(self, wind_speed):
        """
        Calculate risk based on wind conditions
        """
        risk_score = 0.0
        
        # High wind (physical damage, evapotranspiration stress)
        if wind_speed > 20:  # m/s
            risk_score += 0.3
        elif wind_speed > 10:
            risk_score += 0.15
        
        return min(risk_score, 1.0)
    
    def calculate_soil_risk(self, soil_moisture, soil_temperature):
        """
        Calculate risk based on soil conditions
        """
        risk_score = 0.0
        
        # Water stress
        if soil_moisture < 15:  # %
            risk_score += 0.3
        elif soil_moisture < 25:
            risk_score += 0.15
        
        # Soil temperature extremes
        if soil_temperature > 30 or soil_temperature < 5:
            risk_score += 0.2
        
        return min(risk_score, 1.0)
```

## Spectral Risk Assessment

### Vegetation Index-Based Risk

Vegetation indices provide quantitative measures of plant health:

```python
def calculate_spectral_risk(vegetation_indices, anomaly_scores):
    """
    Calculate risk based on spectral analysis
    """
    risk_score = 0.0
    
    # NDVI decline (primary indicator)
    ndvi = vegetation_indices.get('NDVI', 0.8)
    if ndvi < 0.3:
        risk_score += 0.4
    elif ndvi < 0.5:
        risk_score += 0.25
    elif ndvi < 0.6:
        risk_score += 0.1
    
    # EVI decline (more sensitive to canopy)
    evi = vegetation_indices.get('EVI', 0.6)
    if evi < 0.2:
        risk_score += 0.3
    elif evi < 0.4:
        risk_score += 0.15
    
    # Anomaly detection
    if len(anomaly_scores) > 0:
        anomaly_percentage = np.mean(anomaly_scores > 0.7)
        risk_score += anomaly_percentage * 0.3
    
    return min(risk_score, 1.0)
```

### Health Classification Risk

Using the health classification results to assess risk:

```python
def calculate_health_classification_risk(health_map):
    """
    Calculate risk based on health classification results
    """
    if health_map is None:
        return 0.0
    
    # Calculate percentage of each health class
    total_pixels = health_map.size
    diseased_pixels = np.sum(health_map == 3)  # Severe stress/disease
    moderate_stress_pixels = np.sum(health_map == 2)
    mild_stress_pixels = np.sum(health_map == 1)
    
    # Risk calculation based on health distribution
    diseased_risk = (diseased_pixels / total_pixels) * 0.6
    moderate_risk = (moderate_stress_pixels / total_pixels) * 0.3
    mild_risk = (mild_stress_pixels / total_pixels) * 0.1
    
    health_risk = diseased_risk + moderate_risk + mild_risk
    return min(health_risk, 1.0)
```

## Temporal Risk Assessment

### Progression Rate Analysis

Tracking how crop health changes over time:

```python
class TemporalRiskEngine:
    def calculate_progression_risk(self, historical_data):
        """
        Calculate risk based on disease progression rates
        """
        if len(historical_data) < 2:
            return 0.0
        
        # Calculate progression rate
        initial_health = historical_data[0]['average_health']
        current_health = historical_data[-1]['average_health']
        time_span = historical_data[-1]['timestamp'] - historical_data[0]['timestamp']
        
        # Health decline rate (negative values indicate decline)
        decline_rate = (current_health - initial_health) / time_span.days
        
        # Risk based on decline rate
        if decline_rate < -0.1:  # Rapid decline
            return 0.8
        elif decline_rate < -0.05:  # Moderate decline
            return 0.5
        elif decline_rate < -0.01:  # Slow decline
            return 0.2
        else:  # Stable or improving
            return 0.0
    
    def calculate_seasonal_risk(self, current_date, crop_type):
        """
        Calculate risk based on seasonal patterns
        """
        # Day of year
        day_of_year = current_date.timetuple().tm_yday
        
        # Different crops have different risk periods
        if crop_type == 'wheat':
            # Higher risk during flowering and grain filling periods
            if 120 <= day_of_year <= 180:  # May-June
                return 0.4
            elif 210 <= day_of_year <= 270:  # August-September
                return 0.3
        elif crop_type == 'corn':
            # Higher risk during tasseling and silking
            if 180 <= day_of_year <= 240:  # June-July
                return 0.5
        elif crop_type == 'soybeans':
            # Higher risk during flowering and pod filling
            if 210 <= day_of_year <= 270:  # July-August
                return 0.4
        
        return 0.1  # Base risk
```

## Comprehensive Risk Map Generation

### Pixel-Level Risk Calculation

Generating risk maps with spatial resolution:

```python
class RiskAssessmentEngine:
    def __init__(self):
        self.environmental_engine = EnvironmentalRiskEngine()
        self.temporal_engine = TemporalRiskEngine()
    
    def generate_risk_map(self, field_data):
        """
        Generate comprehensive risk map for the field
        """
        height, width = field_data['spatial_dims']
        risk_map = np.zeros((height, width))
        
        # Get environmental risk (assumed constant across field for simplicity)
        environmental_risk = self.environmental_engine.calculate_environmental_risk(
            field_data.get('sensor_data', {})
        )
        
        # Calculate risk for each pixel
        for i in range(height):
            for j in range(width):
                # Extract pixel-specific data
                pixel_data = self.extract_pixel_data(field_data, i, j)
                
                # Calculate individual risk factors
                spectral_risk = self.calculate_spectral_risk(pixel_data)
                temporal_risk = self.calculate_temporal_risk(pixel_data)
                
                # Combine risks with weights
                pixel_risk = (
                    environmental_risk * 0.3 +
                    spectral_risk * 0.5 +
                    temporal_risk * 0.2
                )
                
                risk_map[i, j] = pixel_risk
        
        return risk_map
    
    def extract_pixel_data(self, field_data, i, j):
        """
        Extract data for a specific pixel
        """
        pixel_data = {
            'spectral': field_data.get('spectral_data', np.zeros((224, 128, 128)))[:, i, j],
            'environmental': field_data.get('sensor_data', {}),
            'temporal': field_data.get('temporal_data', {})
        }
        return pixel_data
    
    def calculate_spectral_risk(self, pixel_data):
        """
        Calculate spectral risk for a pixel
        """
        # Simplified calculation - in practice, this would use actual vegetation indices
        spectral_values = pixel_data['spectral']
        nir_mean = np.mean(spectral_values[100:150])  # NIR bands
        red_mean = np.mean(spectral_values[30:40])    # Red bands
        
        # Simple NDVI-like calculation
        ndvi = (nir_mean - red_mean) / (nir_mean + red_mean + 1e-8)
        
        # Convert NDVI to risk (lower NDVI = higher risk)
        if ndvi < 0.3:
            return 0.8
        elif ndvi < 0.5:
            return 0.5
        elif ndvi < 0.7:
            return 0.2
        else:
            return 0.0
    
    def calculate_temporal_risk(self, pixel_data):
        """
        Calculate temporal risk for a pixel
        """
        temporal_data = pixel_data.get('temporal', {})
        # In practice, this would analyze historical data for the pixel
        return 0.1  # Placeholder
```

## Alert Generation

### Risk-Based Alert System

Generating actionable alerts based on risk assessments:

```python
def generate_alerts(self, risk_map, thresholds=None):
    """
    Generate actionable alerts based on risk assessment
    """
    if thresholds is None:
        thresholds = {
            'high_risk': 0.7,
            'medium_risk': 0.4,
            'low_risk': 0.1
        }
    
    alerts = []
    
    # Find high-risk areas
    high_risk_areas = np.where(risk_map > thresholds['high_risk'])
    if len(high_risk_areas[0]) > 0:
        alerts.append({
            'level': 'CRITICAL',
            'message': f'High risk detected in {len(high_risk_areas[0])} pixels',
            'coordinates': list(zip(high_risk_areas[0], high_risk_areas[1])),
            'recommended_action': 'Immediate inspection and treatment required',
            'priority': 1,
            'estimated_loss': self._estimate_potential_loss(len(high_risk_areas[0]), 'high')
        })
    
    # Find medium-risk areas
    medium_risk_mask = (risk_map > thresholds['medium_risk']) & (risk_map <= thresholds['high_risk'])
    medium_risk_areas = np.where(medium_risk_mask)
    if len(medium_risk_areas[0]) > 0:
        alerts.append({
            'level': 'WARNING',
            'message': f'Medium risk detected in {len(medium_risk_areas[0])} pixels',
            'coordinates': list(zip(medium_risk_areas[0], medium_risk_areas[1])),
            'recommended_action': 'Schedule inspection within 48 hours',
            'priority': 2,
            'estimated_loss': self._estimate_potential_loss(len(medium_risk_areas[0]), 'medium')
        })
    
    # Find low-risk areas (informational)
    low_risk_mask = (risk_map > thresholds['low_risk']) & (risk_map <= thresholds['medium_risk'])
    low_risk_areas = np.where(low_risk_mask)
    if len(low_risk_areas[0]) > 0:
        alerts.append({
            'level': 'INFO',
            'message': f'Low risk detected in {len(low_risk_areas[0])} pixels',
            'coordinates': list(zip(low_risk_areas[0], low_risk_areas[1])),
            'recommended_action': 'Continue monitoring',
            'priority': 3
        })
    
    return alerts

def _estimate_potential_loss(self, affected_pixels, risk_level):
    """
    Estimate potential economic loss
    """
    # Simplified loss estimation
    pixel_to_area = 1.0  # 1 pixel = 1 m²
    crop_value_per_m2 = 2.5  # $2.5 per m²
    
    if risk_level == 'high':
        loss_percentage = 0.7  # 70% loss
    elif risk_level == 'medium':
        loss_percentage = 0.3  # 30% loss
    else:
        loss_percentage = 0.1  # 10% loss
    
    total_area = affected_pixels * pixel_to_area
    potential_loss = total_area * crop_value_per_m2 * loss_percentage
    
    return {
        'affected_area_m2': total_area,
        'potential_loss_usd': round(potential_loss, 2),
        'loss_percentage': loss_percentage * 100
    }
```

## Predictive Modeling

### Disease Progression Prediction

Using temporal data to predict future conditions:

```python
class DiseaseProgressionPredictor:
    def predict_progression(self, time_series_data, forecast_days=7):
        """
        Predict disease progression over time
        """
        if len(time_series_data) < 10:
            # Not enough data for reliable prediction
            return self.simple_extrapolation(time_series_data, forecast_days)
        
        # In practice, this would use LSTM or other time series models
        # For this example, we'll use a simple trend-based approach
        return self.trend_based_prediction(time_series_data, forecast_days)
    
    def simple_extrapolation(self, data, forecast_days):
        """
        Simple linear extrapolation
        """
        if len(data) < 2:
            return [data[-1]] * forecast_days if len(data) > 0 else [0.5] * forecast_days
        
        # Calculate trend
        trend = (data[-1] - data[-2]) if len(data) >= 2 else 0
        
        # Generate predictions
        predictions = []
        last_value = data[-1]
        for i in range(forecast_days):
            next_value = last_value + trend * (i + 1)
            # Clamp between 0 and 1
            predictions.append(max(0, min(1, next_value)))
        
        return predictions
    
    def trend_based_prediction(self, data, forecast_days):
        """
        More sophisticated trend-based prediction
        """
        # Calculate moving average trend
        window_size = min(5, len(data))
        if window_size < 2:
            return [data[-1]] * forecast_days
        
        recent_data = data[-window_size:]
        trend = (recent_data[-1] - recent_data[0]) / (window_size - 1)
        
        # Apply dampening factor for long-term predictions
        predictions = []
        last_value = data[-1]
        dampening = 0.95  # Reduces trend impact over time
        
        for i in range(forecast_days):
            trend_factor = trend * (dampening ** i)
            next_value = last_value + trend_factor
            # Clamp between 0 and 1
            predictions.append(max(0, min(1, next_value)))
            last_value = next_value
        
        return predictions
```

## Practical Exercises

### Exercise 1: Implementing a Risk Assessment Engine

```python
import numpy as np
from datetime import datetime, timedelta

class SimpleRiskAssessmentEngine:
    def __init__(self):
        print("Initializing Risk Assessment Engine...")
    
    def calculate_comprehensive_risk(self, ndvi_map, environmental_data):
        """
        Calculate comprehensive risk based on NDVI and environmental data
        """
        print("Calculating comprehensive risk...")
        
        # Calculate spectral risk from NDVI map
        spectral_risk = self.calculate_spectral_risk(ndvi_map)
        print(f"Spectral risk calculated: {spectral_risk:.3f}")
        
        # Calculate environmental risk
        environmental_risk = self.calculate_environmental_risk(environmental_data)
        print(f"Environmental risk calculated: {environmental_risk:.3f}")
        
        # Combine risks (weighted average)
        comprehensive_risk = spectral_risk * 0.6 + environmental_risk * 0.4
        print(f"Comprehensive risk: {comprehensive_risk:.3f}")
        
        return comprehensive_risk
    
    def calculate_spectral_risk(self, ndvi_map):
        """
        Calculate risk based on NDVI values
        """
        # Average NDVI across the field
        avg_ndvi = np.mean(ndvi_map)
        
        # Convert NDVI to risk (lower NDVI = higher risk)
        # NDVI range: -1 to 1, Risk range: 0 to 1
        spectral_risk = 1.0 - ((avg_ndvi + 1.0) / 2.0)
        return max(0, min(1, spectral_risk))  # Clamp to 0-1 range
    
    def calculate_environmental_risk(self, env_data):
        """
        Calculate risk based on environmental conditions
        """
        risk_score = 0.0
        
        # Temperature risk
        temp = env_data.get('temperature', 20)
        if temp > 35 or temp < 5:
            risk_score += 0.4
        elif temp > 30 or temp < 10:
            risk_score += 0.2
        
        # Humidity risk
        humidity = env_data.get('humidity', 50)
        if humidity > 80:
            risk_score += 0.3
        elif humidity > 60:
            risk_score += 0.15
        
        # Rainfall risk
        rainfall = env_data.get('rainfall', 0)
        if rainfall > 20:
            risk_score += 0.2
        elif rainfall > 5:
            risk_score += 0.1
        
        return min(risk_score, 1.0)
    
    def generate_risk_map(self, ndvi_map, env_risk):
        """
        Generate pixel-level risk map
        """
        height, width = ndvi_map.shape
        risk_map = np.zeros((height, width))
        
        # Convert NDVI to pixel-level risks
        # Invert NDVI (lower values = higher risk)
        normalized_ndvi = (ndvi_map + 1.0) / 2.0  # Normalize to 0-1
        ndvi_risk = 1.0 - normalized_ndvi  # Invert for risk
        
        # Combine with environmental risk
        for i in range(height):
            for j in range(width):
                # Weighted combination
                risk_map[i, j] = ndvi_risk[i, j] * 0.7 + env_risk * 0.3
        
        return np.clip(risk_map, 0, 1)  # Ensure values are in 0-1 range

# Example usage
engine = SimpleRiskAssessmentEngine()

# Create sample NDVI map (64x64 pixels)
sample_ndvi = np.random.normal(0.6, 0.2, (64, 64))
sample_ndvi = np.clip(sample_ndvi, -1, 1)  # Ensure valid NDVI range

# Sample environmental data
sample_env = {
    'temperature': 28.5,
    'humidity': 72,
    'rainfall': 5.2
}

# Calculate comprehensive risk
comprehensive_risk = engine.calculate_comprehensive_risk(sample_ndvi, sample_env)
print(f"\nComprehensive field risk: {comprehensive_risk:.3f}")

# Generate risk map
env_risk = engine.calculate_environmental_risk(sample_env)
risk_map = engine.generate_risk_map(sample_ndvi, env_risk)
print(f"Risk map generated with shape: {risk_map.shape}")
print(f"Risk map statistics - Min: {risk_map.min():.3f}, Max: {risk_map.max():.3f}, Mean: {risk_map.mean():.3f}")
```

### Exercise 2: Alert Generation System

```python
def generate_alerts_from_risk_map(risk_map, field_id="Unknown Field"):
    """
    Generate alerts based on risk map analysis
    """
    alerts = []
    
    # Calculate statistics
    avg_risk = np.mean(risk_map)
    max_risk = np.max(risk_map)
    high_risk_pixels = np.sum(risk_map > 0.7)
    medium_risk_pixels = np.sum((risk_map > 0.4) & (risk_map <= 0.7))
    
    print(f"Risk Analysis for {field_id}:")
    print(f"  Average Risk: {avg_risk:.3f}")
    print(f"  Maximum Risk: {max_risk:.3f}")
    print(f"  High Risk Pixels (>0.7): {high_risk_pixels}")
    print(f"  Medium Risk Pixels (0.4-0.7): {medium_risk_pixels}")
    
    # Generate alerts based on risk levels
    if high_risk_pixels > 0:
        # Find coordinates of high-risk areas
        high_risk_coords = np.where(risk_map > 0.7)
        alerts.append({
            'level': 'CRITICAL',
            'field_id': field_id,
            'message': f'Critical risk detected in {high_risk_pixels} areas',
            'coordinates': list(zip(high_risk_coords[0][:5], high_risk_coords[1][:5])),  # First 5 coordinates
            'recommended_action': 'Immediate field inspection and treatment required',
            'priority': 1
        })
    
    if medium_risk_pixels > 0:
        medium_risk_coords = np.where((risk_map > 0.4) & (risk_map <= 0.7))
        alerts.append({
            'level': 'WARNING',
            'field_id': field_id,
            'message': f'Moderate risk detected in {medium_risk_pixels} areas',
            'coordinates': list(zip(medium_risk_coords[0][:5], medium_risk_coords[1][:5])),  # First 5 coordinates
            'recommended_action': 'Schedule field inspection within 48 hours',
            'priority': 2
        })
    
    if len(alerts) == 0:
        alerts.append({
            'level': 'INFO',
            'field_id': field_id,
            'message': 'Field health is good. Continue regular monitoring.',
            'recommended_action': 'Maintain current monitoring schedule',
            'priority': 3
        })
    
    return alerts

# Example usage with the risk map from previous exercise
alerts = generate_alerts_from_risk_map(risk_map, "Sample Field A")

print("\nGenerated Alerts:")
for i, alert in enumerate(alerts, 1):
    print(f"\nAlert {i}:")
    print(f"  Level: {alert['level']}")
    print(f"  Message: {alert['message']}")
    print(f"  Recommended Action: {alert['recommended_action']}")
    if 'coordinates' in alert:
        print(f"  Sample Coordinates: {alert['coordinates'][:3]}...")
```

## Discussion Questions

1. How do different environmental factors contribute to overall crop risk, and why are they weighted differently?
2. What are the challenges in combining spectral and environmental risk factors into a single risk assessment?
3. How can temporal analysis improve the accuracy of risk predictions?
4. What factors should be considered when determining appropriate risk thresholds for alerts?

## Additional Resources

- FAO guidelines on agricultural risk assessment
- Research papers on crop disease prediction models
- Climate risk assessment methodologies
- Machine learning approaches to time series forecasting

## Summary

This chapter covered the risk assessment and prediction capabilities of the AI-Powered Spectral Health Mapping System. We explored how environmental, spectral, and temporal factors are combined to generate comprehensive risk assessments. We also discussed predictive modeling techniques for forecasting crop health trends and generating actionable alerts. Understanding these risk assessment methods is crucial for interpreting system outputs and making informed agricultural decisions, which we'll explore further in the next chapter on visualization and decision support.