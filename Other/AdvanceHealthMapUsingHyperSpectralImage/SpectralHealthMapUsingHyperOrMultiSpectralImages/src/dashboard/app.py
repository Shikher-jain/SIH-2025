import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import dash_bootstrap_components as dbc
from typing import Dict, List
import logging

class SpectralHealthDashboard:
    """Interactive dashboard for spectral health monitoring"""
    
    def __init__(self, data_processor=None, models=None, analytics=None):
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.data_processor = data_processor
        self.models = models
        self.analytics = analytics
        self.logger = logging.getLogger(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("🌱 AI-Powered Spectral Health Mapping System", 
                           className="text-center mb-4"),
                    html.P("Advanced crop health monitoring using hyperspectral imaging and deep learning",
                          className="text-center text-muted mb-4"),
                    html.Hr()
                ])
            ]),
            
            # Control Panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Control Panel"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Field Selection:"),
                                    dcc.Dropdown(
                                        id='field-dropdown',
                                        options=[
                                            {'label': 'Field A - Wheat (120 ha)', 'value': 'field_a'},
                                            {'label': 'Field B - Corn (85 ha)', 'value': 'field_b'},
                                            {'label': 'Field C - Soybeans (200 ha)', 'value': 'field_c'}
                                        ],
                                        value='field_a'
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Analysis Type:"),
                                    dcc.Dropdown(
                                        id='analysis-type',
                                        options=[
                                            {'label': 'Health Status', 'value': 'health'},
                                            {'label': 'Disease Detection', 'value': 'disease'},
                                            {'label': 'Stress Analysis', 'value': 'stress'},
                                            {'label': 'Risk Assessment', 'value': 'risk'}
                                        ],
                                        value='health'
                                    )
                                ], width=4),
                                dbc.Col([
                                    html.Label("Date Range:"),
                                    dcc.DatePickerRange(
                                        id='date-picker',
                                        start_date=datetime.now() - timedelta(days=30),
                                        end_date=datetime.now(),
                                        display_format='YYYY-MM-DD'
                                    )
                                ], width=4)
                            ]),
                            html.Br(),
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button("🔄 Refresh Data", id="refresh-btn", 
                                             color="primary", className="me-2"),
                                    dbc.Button("🚨 Run AI Analysis", id="ai-analysis-btn", 
                                             color="success", className="me-2"),
                                    dbc.Button("📊 Generate Report", id="report-btn", 
                                             color="info")
                                ])
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Real-time Status Cards
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🟢 Healthy", className="card-title text-success"),
                            html.H2(id="healthy-count", children="85.2%", 
                                   className="text-success"),
                            html.P("↑ 2.1% from last week", className="text-muted small")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🟡 Stressed", className="card-title text-warning"),
                            html.H2(id="stressed-count", children="9.8%", 
                                   className="text-warning"),
                            html.P("↓ 1.2% from last week", className="text-muted small")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("🔴 Diseased", className="card-title text-danger"),
                            html.H2(id="diseased-count", children="3.5%", 
                                   className="text-danger"),
                            html.P("↑ 0.8% from last week", className="text-muted small")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚠️ High Risk", className="card-title text-info"),
                            html.H2(id="risk-count", children="1.5%", 
                                   className="text-info"),
                            html.P("New detection", className="text-muted small")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Main Visualization Area
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("🗺️ AI Health Map", className="mb-0"),
                            dbc.Badge("Live", color="success", className="ms-2")
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="health-map", style={'height': '500px'}),
                            dbc.Row([
                                dbc.Col([
                                    html.Small("Click on map for detailed analysis", 
                                             className="text-muted")
                                ], width=6),
                                dbc.Col([
                                    html.Div(id="map-click-info", className="text-end")
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📈 Temporal Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="temporal-trends", style={'height': '240px'}),
                            html.Hr(),
                            html.H6("🔮 7-Day Forecast"),
                            dcc.Graph(id="prediction-chart", style={'height': '200px'})
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Detailed Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔬 Spectral Signature Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="spectral-signature", style={'height': '350px'})
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🌡️ Environmental Conditions"),
                        dbc.CardBody([
                            dcc.Graph(id="environmental-data", style={'height': '350px'})
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # AI Insights and Alerts
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("🚨 Active Alerts", className="mb-0"),
                            dbc.Badge(id="alert-count", children="3", color="danger", className="ms-2")
                        ]),
                        dbc.CardBody([
                            html.Div(id="alerts-container")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("🤖 AI Recommendations", className="mb-0"),
                            dbc.Badge("Powered by Deep Learning", color="info", className="ms-2")
                        ]),
                        dbc.CardBody([
                            html.Div(id="recommendations-container")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Model Performance Metrics
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 AI Model Performance"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Disease Detection"),
                                    dbc.Progress(value=94.2, label="94.2% Accuracy", color="success")
                                ], width=3),
                                dbc.Col([
                                    html.H6("Anomaly Detection"),
                                    dbc.Progress(value=89.7, label="89.7% Precision", color="info")
                                ], width=3),
                                dbc.Col([
                                    html.H6("Segmentation"),
                                    dbc.Progress(value=92.1, label="92.1% IoU Score", color="warning")
                                ], width=3),
                                dbc.Col([
                                    html.H6("Risk Prediction"),
                                    dbc.Progress(value=87.8, label="87.8% F1-Score", color="danger")
                                ], width=3)
                            ])
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P("AI-Powered Spectral Health Mapping System | Last updated: " + 
                          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          className="text-center text-muted")
                ])
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('health-map', 'figure'),
             Output('map-click-info', 'children')],
            [Input('field-dropdown', 'value'),
             Input('analysis-type', 'value'),
             Input('refresh-btn', 'n_clicks'),
             Input('ai-analysis-btn', 'n_clicks')]
        )
        def update_health_map(field_id, analysis_type, refresh_clicks, ai_clicks):
            """Update the main health map visualization"""
            # Generate sample health map data based on analysis type
            health_data = self.generate_sample_health_map(analysis_type)
            
            if analysis_type == 'health':
                colorscale = [
                    [0, 'red'],      # Diseased
                    [0.3, 'orange'], # Stressed
                    [0.7, 'yellow'], # Moderate
                    [1, 'green']     # Healthy
                ]
                title = "Crop Health Status Map"
                colorbar_title = "Health Score"
            elif analysis_type == 'disease':
                colorscale = 'Reds'
                title = "Disease Probability Map"
                colorbar_title = "Disease Probability"
            elif analysis_type == 'stress':
                colorscale = 'YlOrRd'
                title = "Plant Stress Map"
                colorbar_title = "Stress Level"
            else:  # risk
                colorscale = 'Plasma'
                title = "Risk Assessment Map"
                colorbar_title = "Risk Score"
            
            fig = go.Figure(data=go.Heatmap(
                z=health_data,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title)
            ))
            
            fig.update_layout(
                title=title,
                xaxis_title="East-West (pixels)",
                yaxis_title="North-South (pixels)",
                height=500
            )
            
            click_info = html.Small("Map updated", className="text-success")
            
            return fig, click_info
        
        @self.app.callback(
            Output('temporal-trends', 'figure'),
            [Input('field-dropdown', 'value')]
        )
        def update_temporal_trends(field_id):
            """Update temporal health trends"""
            try:
                dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
                
                # Generate realistic health trend data
                base_health = 0.8
                seasonal_variation = 0.1 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
                noise = np.random.normal(0, 0.05, len(dates))
                
                # Add some disease events
                disease_events = [100, 180, 250]  # Day of year
                for event_day in disease_events:
                    if event_day < len(dates):
                        noise[event_day:event_day+14] -= 0.2
                
                health_scores = np.clip(base_health + seasonal_variation + noise, 0, 1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=health_scores,
                    mode='lines',
                    name='Average Health Score',
                    line=dict(color='green', width=2)
                ))
                
                # Add disease event markers
                for event_day in disease_events:
                    if event_day < len(dates):
                        fig.add_vline(x=dates[event_day], line_dash="dash", 
                                     line_color="red", annotation_text="Disease Event")
                
                fig.update_layout(
                    title="Health Trend Over Time",
                    xaxis_title="Date",
                    yaxis_title="Health Score",
                    height=240,
                    margin=dict(t=40, b=40, l=40, r=40)
                )
                
                return fig
            except Exception as e:
                # Return a simple fallback chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[0.8, 0.75, 0.7, 0.8, 0.85],
                    mode='lines',
                    name='Health Score',
                    line=dict(color='green', width=2)
                ))
                fig.update_layout(
                    title="Health Trend (Fallback)",
                    xaxis_title="Time Period",
                    yaxis_title="Health Score",
                    height=240
                )
                return fig
        
        @self.app.callback(
            Output('prediction-chart', 'figure'),
            [Input('field-dropdown', 'value')]
        )
        def update_prediction_chart(field_id):
            """Update 7-day prediction chart"""
            future_dates = pd.date_range(start=datetime.now(), periods=7, freq='D')
            
            # Simulate AI predictions
            current_health = 0.75
            predictions = []
            for i in range(7):
                # Add slight decline trend with some randomness
                pred = current_health - (i * 0.02) + np.random.normal(0, 0.01)
                predictions.append(max(0.3, min(1.0, pred)))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted Health',
                line=dict(color='blue', width=2, dash='dot'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="AI Health Forecast",
                xaxis_title="Date",
                yaxis_title="Predicted Health",
                height=200,
                margin=dict(t=40, b=40, l=40, r=40)
            )
            
            return fig
        
        @self.app.callback(
            Output('spectral-signature', 'figure'),
            [Input('health-map', 'clickData'),
             Input('field-dropdown', 'value')]
        )
        def update_spectral_signature(click_data, field_id):
            """Update spectral signature for selected pixel"""
            wavelengths = np.linspace(400, 2500, 224)
            
            if click_data:
                # Simulate different spectral signatures based on click location
                x = click_data['points'][0]['x']
                y = click_data['points'][0]['y']
                health_level = np.random.random()
                
                if health_level > 0.7:
                    signature = self.generate_healthy_signature(wavelengths)
                    title = f"Healthy Vegetation Signature (Pixel: {x}, {y})"
                    color = 'green'
                elif health_level > 0.4:
                    signature = self.generate_stressed_signature(wavelengths)
                    title = f"Stressed Vegetation Signature (Pixel: {x}, {y})"
                    color = 'orange'
                else:
                    signature = self.generate_diseased_signature(wavelengths)
                    title = f"Diseased Vegetation Signature (Pixel: {x}, {y})"
                    color = 'red'
            else:
                signature = self.generate_healthy_signature(wavelengths)
                title = "Average Spectral Signature"
                color = 'blue'
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wavelengths,
                y=signature,
                mode='lines',
                name='Reflectance',
                line=dict(color=color, width=2)
            ))
            
            # Add important spectral regions
            fig.add_vrect(x0=400, x1=700, fillcolor="lightblue", opacity=0.2, 
                         annotation_text="Visible", annotation_position="top left")
            fig.add_vrect(x0=700, x1=1300, fillcolor="lightgreen", opacity=0.2, 
                         annotation_text="NIR", annotation_position="top left")
            fig.add_vrect(x0=1300, x1=2500, fillcolor="lightyellow", opacity=0.2, 
                         annotation_text="SWIR", annotation_position="top left")
            
            fig.update_layout(
                title=title,
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                height=350
            )
            
            return fig
        
        @self.app.callback(
            Output('environmental-data', 'figure'),
            [Input('field-dropdown', 'value')]
        )
        def update_environmental_data(field_id):
            """Update environmental conditions chart"""
            try:
                # Generate sample environmental data  
                dates = pd.date_range(start=datetime.now() - timedelta(days=7), 
                                     end=datetime.now(), freq='h')  # Changed from 'H' to 'h'
                
                temperature = 20 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 24) + \
                             np.random.normal(0, 2, len(dates))
                humidity = 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 24 + np.pi) + \
                          np.random.normal(0, 5, len(dates))
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=dates, y=temperature, name='Temperature (°C)',
                    line=dict(color='red'), yaxis='y'
                ))
                
                fig.add_trace(go.Scatter(
                    x=dates, y=humidity, name='Humidity (%)',
                    line=dict(color='blue'), yaxis='y2'
                ))
                
                fig.update_layout(
                    title="Environmental Conditions (Last 7 Days)",
                    xaxis_title="Time",
                    yaxis=dict(title="Temperature (°C)", side="left"),
                    yaxis2=dict(title="Humidity (%)", side="right", overlaying="y"),
                    height=350
                )
                
                return fig
            except Exception as e:
                # Return fallback chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[25, 23, 28, 26, 24],
                    name='Temperature (°C)',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title="Environmental Conditions (Fallback)",
                    xaxis_title="Time Period",
                    yaxis_title="Temperature (°C)",
                    height=350
                )
                return fig
        
        @self.app.callback(
            Output('alerts-container', 'children'),
            [Input('ai-analysis-btn', 'n_clicks')]
        )
        def update_alerts(n_clicks):
            """Update alerts display"""
            alerts = [
                {
                    "level": "CRITICAL", 
                    "message": "Fungal disease outbreak detected in Zone A3 (12.3 ha)", 
                    "time": "2 minutes ago",
                    "action": "Immediate fungicide treatment required",
                    "confidence": 94.2
                },
                {
                    "level": "WARNING", 
                    "message": "Water stress indicators in Zone B2 (5.7 ha)", 
                    "time": "15 minutes ago",
                    "action": "Increase irrigation within 24 hours",
                    "confidence": 87.5
                },
                {
                    "level": "INFO", 
                    "message": "Optimal growing conditions detected in Zone C1", 
                    "time": "1 hour ago",
                    "action": "Continue current management practices",
                    "confidence": 92.1
                }
            ]
            
            alert_components = []
            for alert in alerts:
                if alert["level"] == "CRITICAL":
                    color = "danger"
                    icon = "🚨"
                elif alert["level"] == "WARNING":
                    color = "warning"
                    icon = "⚠️"
                else:
                    color = "info"
                    icon = "ℹ️"
                
                alert_components.append(
                    dbc.Alert([
                        html.H6([icon, f' {alert["level"]}'], className="alert-heading"),
                        html.P(alert["message"], className="mb-1"),
                        html.Small([
                            html.Strong("Action: "), alert["action"], html.Br(),
                            html.Strong("Confidence: "), f'{alert["confidence"]}%', html.Br(),
                            html.Strong("Time: "), alert["time"]
                        ], className="text-muted")
                    ], color=color, className="mb-2")
                )
            
            return alert_components
        
        @self.app.callback(
            Output('recommendations-container', 'children'),
            [Input('ai-analysis-btn', 'n_clicks')]
        )
        def update_recommendations(n_clicks):
            """Update AI recommendations"""
            recommendations = [
                {
                    "title": "Precision Fungicide Application",
                    "description": "Apply copper-based fungicide to affected areas in Zone A3",
                    "priority": "High",
                    "cost": "$245/ha",
                    "effectiveness": "85%",
                    "timeframe": "Next 6 hours"
                },
                {
                    "title": "Irrigation Optimization",
                    "description": "Increase irrigation frequency in Zone B2 from 2 to 3 times per week",
                    "priority": "Medium",
                    "cost": "$45/ha",
                    "effectiveness": "92%",
                    "timeframe": "Within 24 hours"
                },
                {
                    "title": "Preventive Monitoring",
                    "description": "Enhanced spectral monitoring for early disease detection",
                    "priority": "Low",
                    "cost": "$15/ha",
                    "effectiveness": "78%",
                    "timeframe": "Ongoing"
                }
            ]
            
            recommendation_components = []
            for rec in recommendations:
                if rec["priority"] == "High":
                    color = "danger"
                elif rec["priority"] == "Medium":
                    color = "warning"
                else:
                    color = "success"
                
                recommendation_components.append(
                    dbc.Card([
                        dbc.CardBody([
                            html.H6(rec["title"], className="card-title"),
                            html.P(rec["description"], className="card-text"),
                            dbc.Row([
                                dbc.Col([
                                    html.Small([
                                        html.Strong("Cost: "), rec["cost"], html.Br(),
                                        html.Strong("Effectiveness: "), rec["effectiveness"]
                                    ])
                                ], width=6),
                                dbc.Col([
                                    html.Small([
                                        html.Strong("Timeframe: "), rec["timeframe"], html.Br(),
                                        dbc.Badge(rec["priority"], color=color)
                                    ])
                                ], width=6)
                            ])
                        ])
                    ], className="mb-2")
                )
            
            return recommendation_components
    
    def generate_sample_health_map(self, analysis_type: str = 'health') -> np.ndarray:
        """Generate sample health map data"""
        np.random.seed(42)
        
        if analysis_type == 'health':
            base_map = np.random.random((50, 50)) * 0.3 + 0.7
        elif analysis_type == 'disease':
            base_map = np.random.random((50, 50)) * 0.2
        elif analysis_type == 'stress':
            base_map = np.random.random((50, 50)) * 0.4
        else:  # risk
            base_map = np.random.random((50, 50)) * 0.3
        
        # Add some problematic areas
        problem_centers = [(10, 15), (35, 40), (25, 8)]
        for center in problem_centers:
            y, x = np.ogrid[:50, :50]
            mask = (y - center[0])**2 + (x - center[1])**2 <= 25
            
            if analysis_type == 'health':
                base_map[mask] *= 0.3  # Lower health
            else:
                base_map[mask] = np.clip(base_map[mask] + 0.5, 0, 1)  # Higher problems
        
        return base_map
    
    def generate_healthy_signature(self, wavelengths: np.ndarray) -> np.ndarray:
        """Generate healthy vegetation spectral signature"""
        signature = np.zeros_like(wavelengths, dtype=float)
        
        # Visible region (low reflectance)
        visible_mask = wavelengths < 700
        visible_count = int(np.sum(visible_mask))
        if visible_count > 0:
            signature[visible_mask] = 0.05 + 0.02 * np.random.random(visible_count)
        
        # Green peak
        green_mask = (wavelengths >= 500) & (wavelengths <= 600)
        green_count = int(np.sum(green_mask))
        if green_count > 0:
            signature[green_mask] = 0.12 + 0.03 * np.random.random(green_count)
        
        # NIR plateau (high reflectance)
        nir_mask = (wavelengths >= 700) & (wavelengths <= 1300)
        nir_count = int(np.sum(nir_mask))
        if nir_count > 0:
            signature[nir_mask] = 0.45 + 0.05 * np.random.random(nir_count)
        
        # SWIR (moderate reflectance with water absorption)
        swir_mask = wavelengths > 1300
        swir_count = int(np.sum(swir_mask))
        if swir_count > 0:
            signature[swir_mask] = 0.25 + 0.03 * np.random.random(swir_count)
        
        return signature
    
    def generate_stressed_signature(self, wavelengths: np.ndarray) -> np.ndarray:
        """Generate stressed vegetation spectral signature"""
        signature = self.generate_healthy_signature(wavelengths)
        
        # Reduce NIR reflectance
        nir_mask = (wavelengths >= 700) & (wavelengths <= 1300)
        signature[nir_mask] *= 0.7
        
        # Increase visible reflectance
        visible_mask = wavelengths < 700
        signature[visible_mask] *= 1.3
        
        return signature
    
    def generate_diseased_signature(self, wavelengths: np.ndarray) -> np.ndarray:
        """Generate diseased vegetation spectral signature"""
        signature = self.generate_healthy_signature(wavelengths)
        
        # Significantly reduce NIR reflectance
        nir_mask = (wavelengths >= 700) & (wavelengths <= 1300)
        signature[nir_mask] *= 0.4
        
        # Increase visible reflectance significantly
        visible_mask = wavelengths < 700
        signature[visible_mask] *= 1.8
        
        return signature
    
    def run(self, debug=True, host='127.0.0.1', port=8050):
        """Run the dashboard"""
        self.logger.info(f"Starting dashboard at http://{host}:{port}")
        self.app.run(debug=debug, host=host, port=port)