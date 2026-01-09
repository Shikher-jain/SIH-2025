import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from typing import Dict, List, Tuple
import logging

class DiseaseProgressionPredictor:
    """Predict disease progression and spread patterns"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.progression_model = None
        self.spread_model = None
        self.logger = logging.getLogger(__name__)
        
    def build_progression_model(self, input_features: int):
        """Build model for disease progression prediction"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                keras.layers.LSTM(128, return_sequences=True, input_shape=(None, input_features)),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(64, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(32),
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')  # Progression probability
            ])
            
            model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            self.progression_model = model
            return model
        except ImportError:
            # Fallback to scikit-learn
            self.progression_model = RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
            return self.progression_model
    
    def build_spread_model(self, spatial_features: int):
        """Build model for spatial disease spread prediction"""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
            
            model = keras.Sequential([
                keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', 
                                   input_shape=(None, None, spatial_features)),
                keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
                keras.layers.UpSampling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
                keras.layers.Conv2D(1, (1, 1), activation='sigmoid')  # Spread probability map
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            self.spread_model = model
            return model
        except ImportError:
            # Fallback to traditional ML approach
            self.spread_model = GradientBoostingClassifier(
                n_estimators=100,
                random_state=42
            )
            return self.spread_model
    
    def predict_progression(self, time_series_data: np.ndarray, 
                          forecast_days: int = 7) -> np.ndarray:
        """Predict disease progression over time"""
        if hasattr(self.progression_model, 'predict') and hasattr(self.progression_model, 'layers'):
            # TensorFlow model
            predictions = []
            current_data = time_series_data
            
            for _ in range(forecast_days):
                pred = self.progression_model.predict(current_data[-1:])
                predictions.append(pred[0][0])
                
                # Update input for next prediction
                new_row = np.append(current_data[0, -1, :-1], pred[0][0]).reshape(1, 1, -1)
                current_data = np.concatenate([current_data[:, 1:, :], new_row], axis=1)
            
            return np.array(predictions)
        else:
            # Scikit-learn model
            # Flatten time series for traditional ML
            flattened_data = time_series_data.reshape(time_series_data.shape[0], -1)
            predictions = []
            
            for _ in range(forecast_days):
                pred = self.progression_model.predict(flattened_data[-1:])
                predictions.append(pred[0])
                
                # Simple progression assumption
                flattened_data = np.roll(flattened_data, -1, axis=1)
                flattened_data[0, -1] = pred[0]
            
            return np.array(predictions)
    
    def predict_spread(self, current_map: np.ndarray) -> np.ndarray:
        """Predict spatial spread of disease"""
        if hasattr(self.spread_model, 'predict') and hasattr(self.spread_model, 'layers'):
            # TensorFlow model
            return self.spread_model.predict(current_map[np.newaxis, ...])[0]
        else:
            # Scikit-learn approach - use neighborhood features
            spread_prob = np.zeros_like(current_map)
            
            for i in range(1, current_map.shape[0] - 1):
                for j in range(1, current_map.shape[1] - 1):
                    # Extract neighborhood features
                    neighborhood = current_map[i-1:i+2, j-1:j+2].flatten()
                    prob = self.spread_model.predict_proba([neighborhood])[0][1]
                    spread_prob[i, j] = prob
            
            return spread_prob

class RiskAssessmentEngine:
    """Comprehensive risk assessment and early warning system"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_factors = {
            'environmental': ['temperature', 'humidity', 'rainfall', 'wind'],
            'biological': ['plant_stress', 'pest_pressure', 'disease_history'],
            'spectral': ['ndvi_decline', 'stress_indicators', 'anomaly_scores'],
            'temporal': ['progression_rate', 'seasonal_patterns']
        }
        self.logger = logging.getLogger(__name__)
        
    def calculate_environmental_risk(self, sensor_data: Dict) -> float:
        """Calculate environmental risk score"""
        risk_score = 0.0
        
        # Temperature stress
        temp = sensor_data.get('air_temperature', 20)
        if temp > 35 or temp < 5:
            risk_score += 0.3
        elif temp > 30 or temp < 10:
            risk_score += 0.1
            
        # Humidity risk (disease favorable conditions)
        humidity = sensor_data.get('humidity', 50)
        if humidity > 85:
            risk_score += 0.25
        elif humidity > 70:
            risk_score += 0.1
            
        # Rainfall impact
        rainfall = sensor_data.get('rainfall', 0)
        if rainfall > 10:  # Heavy rain can spread diseases
            risk_score += 0.2
        elif rainfall < 1:  # Drought stress
            risk_score += 0.15
            
        # Leaf wetness duration
        leaf_wetness = sensor_data.get('leaf_wetness', 0)
        if leaf_wetness > 6:  # Hours of wetness
            risk_score += 0.25
            
        return min(risk_score, 1.0)
    
    def calculate_spectral_risk(self, spectral_indices: Dict, 
                              anomaly_scores: np.ndarray) -> float:
        """Calculate risk based on spectral analysis"""
        risk_score = 0.0
        
        # NDVI decline
        ndvi = spectral_indices.get('NDVI', 0.8)
        if ndvi < 0.3:
            risk_score += 0.4
        elif ndvi < 0.5:
            risk_score += 0.2
            
        # Stress indicators
        if spectral_indices.get('PRI', 0) < -0.1:  # Photochemical stress
            risk_score += 0.2
            
        # Anomaly detection
        anomaly_percentage = np.mean(anomaly_scores > 0.7)
        risk_score += anomaly_percentage * 0.4
        
        return min(risk_score, 1.0)
    
    def calculate_temporal_risk(self, temporal_data: Dict) -> float:
        """Calculate risk based on temporal patterns"""
        risk_score = 0.0
        
        # Disease progression rate
        progression_rate = temporal_data.get('progression_rate', 0)
        if progression_rate > 0.1:  # Fast progression
            risk_score += 0.3
        elif progression_rate > 0.05:
            risk_score += 0.15
        
        # Seasonal factors
        current_season = temporal_data.get('season', 'unknown')
        if current_season in ['spring', 'summer']:  # High disease pressure seasons
            risk_score += 0.1
        
        # Historical disease occurrence
        historical_risk = temporal_data.get('historical_disease_probability', 0)
        risk_score += historical_risk * 0.2
        
        return min(risk_score, 1.0)
    
    def generate_risk_map(self, field_data: Dict) -> np.ndarray:
        """Generate comprehensive risk map for the field"""
        height, width = field_data['spatial_dims']
        risk_map = np.zeros((height, width))
        
        for i in range(height):
            for j in range(width):
                pixel_data = {
                    'spectral': field_data.get('spectral_data', np.zeros((224, height, width)))[:, i, j],
                    'environmental': field_data.get('sensor_data', {}),
                    'temporal': field_data.get('temporal_data', {})
                }
                
                env_risk = self.calculate_environmental_risk(pixel_data['environmental'])
                spec_risk = self.calculate_spectral_risk(
                    field_data.get('vegetation_indices', {}), 
                    field_data.get('anomaly_scores', np.zeros((height, width)))[i, j:j+1]
                )
                temp_risk = self.calculate_temporal_risk(pixel_data['temporal'])
                
                # Combined risk with weights
                risk_map[i, j] = (env_risk * 0.3 + spec_risk * 0.5 + temp_risk * 0.2)
        
        return risk_map
    
    def generate_alerts(self, risk_map: np.ndarray, 
                       thresholds: Dict) -> List[Dict]:
        """Generate actionable alerts based on risk assessment"""
        alerts = []
        
        high_risk_threshold = thresholds.get('high_risk', 0.7)
        medium_risk_threshold = thresholds.get('medium_risk', 0.4)
        
        high_risk_areas = np.where(risk_map > high_risk_threshold)
        medium_risk_areas = np.where(
            (risk_map > medium_risk_threshold) & 
            (risk_map <= high_risk_threshold)
        )
        
        if len(high_risk_areas[0]) > 0:
            alerts.append({
                'level': 'CRITICAL',
                'message': f'High risk detected in {len(high_risk_areas[0])} pixels',
                'coordinates': list(zip(high_risk_areas[0], high_risk_areas[1])),
                'recommended_action': 'Immediate inspection and treatment required',
                'priority': 1,
                'estimated_loss': self._estimate_potential_loss(len(high_risk_areas[0]), 'high')
            })
        
        if len(medium_risk_areas[0]) > 0:
            alerts.append({
                'level': 'WARNING',
                'message': f'Medium risk detected in {len(medium_risk_areas[0])} pixels',
                'coordinates': list(zip(medium_risk_areas[0], medium_risk_areas[1])),
                'recommended_action': 'Schedule inspection within 48 hours',
                'priority': 2,
                'estimated_loss': self._estimate_potential_loss(len(medium_risk_areas[0]), 'medium')
            })
        
        return alerts
    
    def _estimate_potential_loss(self, affected_pixels: int, risk_level: str) -> Dict:
        """Estimate potential economic loss"""
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

class InterventionRecommender:
    """Recommend interventions based on detected problems"""
    
    def __init__(self):
        self.treatment_database = {
            'fungal_disease': {
                'treatments': ['fungicide_spray', 'remove_infected_plants', 'improve_ventilation'],
                'timing': 'immediate',
                'effectiveness': 0.85
            },
            'bacterial_disease': {
                'treatments': ['copper_spray', 'remove_infected_plants', 'soil_treatment'],
                'timing': 'immediate',
                'effectiveness': 0.75
            },
            'pest_damage': {
                'treatments': ['insecticide_spray', 'biological_control', 'crop_rotation'],
                'timing': 'within_24h',
                'effectiveness': 0.80
            },
            'nutrient_stress': {
                'treatments': ['fertilizer_application', 'soil_amendment', 'pH_adjustment'],
                'timing': 'within_week',
                'effectiveness': 0.90
            },
            'water_stress': {
                'treatments': ['irrigation_adjustment', 'mulching', 'soil_improvement'],
                'timing': 'immediate',
                'effectiveness': 0.95
            }
        }
    
    def recommend_intervention(self, problem_type: str, severity: float, 
                             field_conditions: Dict) -> Dict:
        """Recommend specific interventions"""
        if problem_type not in self.treatment_database:
            return {'error': f'Unknown problem type: {problem_type}'}
        
        treatment_info = self.treatment_database[problem_type]
        
        # Adjust recommendations based on severity
        if severity > 0.8:
            urgency = 'immediate'
            intensity = 'high'
        elif severity > 0.5:
            urgency = 'within_24h'
            intensity = 'medium'
        else:
            urgency = 'within_week'
            intensity = 'low'
        
        recommendations = {
            'problem_type': problem_type,
            'severity': severity,
            'urgency': urgency,
            'intensity': intensity,
            'treatments': treatment_info['treatments'],
            'expected_effectiveness': treatment_info['effectiveness'],
            'estimated_cost': self._estimate_treatment_cost(
                treatment_info['treatments'], intensity, field_conditions
            ),
            'monitoring_schedule': self._create_monitoring_schedule(urgency)
        }
        
        return recommendations
    
    def _estimate_treatment_cost(self, treatments: List[str], intensity: str, 
                               field_conditions: Dict) -> Dict:
        """Estimate treatment costs"""
        cost_multipliers = {'low': 0.5, 'medium': 1.0, 'high': 1.5}
        base_costs = {
            'fungicide_spray': 50,
            'insecticide_spray': 40,
            'fertilizer_application': 30,
            'irrigation_adjustment': 20,
            'soil_treatment': 80
        }
        
        total_cost = 0
        for treatment in treatments:
            if treatment in base_costs:
                cost = base_costs[treatment] * cost_multipliers[intensity]
                total_cost += cost
        
        field_area = field_conditions.get('area_hectares', 1.0)
        total_cost *= field_area
        
        return {
            'total_cost_usd': round(total_cost, 2),
            'cost_per_hectare': round(total_cost / field_area, 2)
        }
    
    def _create_monitoring_schedule(self, urgency: str) -> List[str]:
        """Create monitoring schedule based on urgency"""
        if urgency == 'immediate':
            return ['daily_for_1_week', 'every_3_days_for_2_weeks', 'weekly_for_1_month']
        elif urgency == 'within_24h':
            return ['every_3_days_for_1_week', 'weekly_for_3_weeks']
        else:
            return ['weekly_for_2_weeks', 'biweekly_for_1_month']