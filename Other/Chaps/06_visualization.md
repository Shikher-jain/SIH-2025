# Chapter 6: Visualization and Decision Support

## Learning Objectives

By the end of this chapter, students will be able to:
- Understand different visualization techniques for agricultural data
- Explain how health maps and risk maps are displayed
- Implement interactive dashboards for crop monitoring
- Interpret system outputs for decision-making
- Create custom visualizations for specific agricultural needs

## Key Concepts
 
### Importance of Visualization in Agriculture

Visualization plays a crucial role in agricultural decision-making by:

1. **Making Complex Data Accessible**: Converting numerical data into intuitive visual representations
2. **Enabling Quick Assessment**: Allowing rapid identification of problem areas
3. **Supporting Spatial Analysis**: Showing geographic distribution of crop health
4. **Facilitating Communication**: Helping farmers and agronomists discuss findings
5. **Enabling Historical Comparison**: Tracking changes over time

### Types of Agricultural Visualizations

The AI-Powered Spectral Health Mapping System generates several types of visualizations:

```
System Visualizations
├── Health Maps
│   ├── Classification Maps
│   ├── Confidence Maps
│   └── Change Detection Maps
├── Risk Maps
│   ├── Environmental Risk Maps
│   ├── Spectral Risk Maps
│   └── Comprehensive Risk Maps
├── Vegetation Indices
│   ├── NDVI Maps
│   ├── EVI Maps
│   └── Custom Index Maps
├── Time Series Visualizations
│   ├── Progression Charts
│   ├── Historical Trends
│   └── Forecast Visualizations
└── Decision Support
    ├── Alert Dashboards
    ├── Recommendation Panels
    └── Treatment Planning Tools
```

## Health Map Visualization

### Classification Maps

Health classification maps show the health status of each pixel in the field:

```python
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

def visualize_health_map(health_map, title="Crop Health Map"):
    """
    Visualize health map with appropriate color coding
    """
    # Define colors for health classes
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']  # Green, Yellow, Orange, Red
    labels = ['Healthy', 'Mild Stress', 'Moderate Stress', 'Severe Stress/Disease']
    
    # Create custom colormap
    cmap = ListedColormap(colors)
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    im = plt.imshow(health_map, cmap=cmap, vmin=0, vmax=3)
    plt.title(title, fontsize=16, pad=20)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=labels[i]) for i in range(4)]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    unique, counts = np.unique(health_map, return_counts=True)
    print("Health Map Statistics:")
    for i, count in enumerate(counts):
        percentage = (count / health_map.size) * 100
        print(f"  {labels[i]}: {count} pixels ({percentage:.1f}%)")

# Example usage
# Create sample health map
sample_health_map = np.random.choice([0, 1, 2, 3], size=(128, 128), p=[0.6, 0.25, 0.1, 0.05])
visualize_health_map(sample_health_map, "Sample Field Health Map")
```

### Continuous Health Visualization

For models that output continuous health scores:

```python
def visualize_continuous_health(health_scores, title="Continuous Health Map"):
    """
    Visualize continuous health scores
    """
    plt.figure(figsize=(12, 10))
    im = plt.imshow(health_scores, cmap='RdYlGn', vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar(im, label='Health Score (1=Healthy, 0=Severely Stressed)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print(f"Health Score Statistics:")
    print(f"  Mean: {np.mean(health_scores):.3f}")
    print(f"  Std Dev: {np.std(health_scores):.3f}")
    print(f"  Min: {np.min(health_scores):.3f}")
    print(f"  Max: {np.max(health_scores):.3f}")

# Example usage
sample_health_scores = np.random.beta(2, 1, (128, 128))  # Beta distribution for realistic scores
visualize_continuous_health(sample_health_scores, "Sample Continuous Health Map")
```

## Risk Map Visualization

### Comprehensive Risk Maps

Risk maps combine multiple factors to show overall risk levels:

```python
def visualize_risk_map(risk_map, title="Field Risk Map"):
    """
    Visualize risk map with gradient colors
    """
    plt.figure(figsize=(12, 10))
    im = plt.imshow(risk_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    plt.title(title, fontsize=16)
    plt.colorbar(im, label='Risk Level (0=Low Risk, 1=High Risk)')
    
    # Add risk level indicators
    risk_levels = [(0.0, 0.3, 'Low', '#2E8B57'), 
                   (0.3, 0.6, 'Moderate', '#FFD700'), 
                   (0.6, 1.0, 'High', '#DC143C')]
    
    for min_risk, max_risk, label, color in risk_levels:
        area_percentage = np.mean((risk_map >= min_risk) & (risk_map < max_risk)) * 100
        print(f"  {label} Risk ({min_risk}-{max_risk}): {area_percentage:.1f}%")
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
sample_risk_map = np.random.beta(1.5, 2, (128, 128))  # Skewed toward lower risk
visualize_risk_map(sample_risk_map, "Sample Field Risk Map")
```

### Multi-Layer Risk Visualization

Showing different risk components:

```python
def visualize_multi_layer_risk(env_risk, spectral_risk, temporal_risk):
    """
    Visualize multiple risk components side by side
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Environmental Risk
    im1 = axes[0].imshow(env_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[0].set_title('Environmental Risk', fontsize=14)
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # Spectral Risk
    im2 = axes[1].imshow(spectral_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[1].set_title('Spectral Risk', fontsize=14)
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # Temporal Risk
    im3 = axes[2].imshow(temporal_risk, cmap='RdYlGn_r', vmin=0, vmax=1)
    axes[2].set_title('Temporal Risk', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    plt.tight_layout()
    plt.show()

# Example usage
env_risk = np.random.beta(1, 2, (128, 128))
spectral_risk = np.random.beta(1.2, 2, (128, 128))
temporal_risk = np.random.beta(1.1, 2, (128, 128))

visualize_multi_layer_risk(env_risk, spectral_risk, temporal_risk)
```

## Vegetation Index Visualization

### NDVI Maps

Visualizing vegetation indices for crop health assessment:

```python
def visualize_vegetation_index(index_map, index_name="Vegetation Index", vmin=-1, vmax=1):
    """
    Visualize vegetation index maps
    """
    plt.figure(figsize=(12, 10))
    im = plt.imshow(index_map, cmap='RdYlGn', vmin=vmin, vmax=vmax)
    plt.title(f'{index_name} Map', fontsize=16)
    plt.colorbar(im, label=f'{index_name} Value')
    
    # Statistics
    print(f"{index_name} Statistics:")
    print(f"  Mean: {np.mean(index_map):.3f}")
    print(f"  Std Dev: {np.std(index_map):.3f}")
    print(f"  Min: {np.min(index_map):.3f}")
    print(f"  Max: {np.max(index_map):.3f}")
    
    # Health categories
    healthy = np.sum(index_map > 0.7)
    stressed = np.sum((index_map > 0.4) & (index_map <= 0.7))
    diseased = np.sum(index_map <= 0.4)
    
    print(f"  Healthy Areas (>0.7): {healthy} pixels ({healthy/index_map.size*100:.1f}%)")
    print(f"  Stressed Areas (0.4-0.7): {stressed} pixels ({stressed/index_map.size*100:.1f}%)")
    print(f"  Diseased Areas (≤0.4): {diseased} pixels ({diseased/index_map.size*100:.1f}%)")
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
sample_ndvi = np.random.normal(0.6, 0.2, (128, 128))
sample_ndvi = np.clip(sample_ndvi, -1, 1)  # Ensure valid range
visualize_vegetation_index(sample_ndvi, "NDVI", vmin=-1, vmax=1)
```

## Time Series Visualization

### Progression Charts

Showing how crop health changes over time:

```python
def plot_health_progression(dates, health_scores, title="Crop Health Progression"):
    """
    Plot health progression over time
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, health_scores, marker='o', linewidth=2, markersize=6)
    plt.title(title, fontsize=16)
    plt.xlabel('Date')
    plt.ylabel('Health Score (1=Healthy, 0=Severely Stressed)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(len(dates)), health_scores, 1)
    p = np.poly1d(z)
    plt.plot(dates, p(range(len(dates))), "r--", alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
    plt.legend()
    
    # Highlight critical points
    min_health_idx = np.argmin(health_scores)
    max_health_idx = np.argmax(health_scores)
    
    plt.annotate(f'Min: {health_scores[min_health_idx]:.2f}', 
                xy=(dates[min_health_idx], health_scores[min_health_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.annotate(f'Max: {health_scores[max_health_idx]:.2f}', 
                xy=(dates[max_health_idx], health_scores[max_health_idx]),
                xytext=(10, -20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.show()

# Example usage
import pandas as pd
from datetime import datetime, timedelta

# Generate sample time series data
start_date = datetime.now() - timedelta(days=30)
dates = [start_date + timedelta(days=i) for i in range(30)]
health_scores = np.random.beta(2 + np.sin(np.linspace(0, 2*np.pi, 30)), 2, 30)

plot_health_progression(dates, health_scores, "30-Day Crop Health Progression")
```

## Interactive Dashboard Implementation

### Web-Based Dashboard with Dash

Creating an interactive dashboard for real-time monitoring:

```python
# Note: This is a conceptual example. In practice, you would need to install dash:
# pip install dash plotly

def create_dashboard_layout():
    """
    Create layout for interactive dashboard (conceptual)
    """
    layout_description = """
    Dashboard Layout Components:
    
    1. Header
       - System title and logo
       - Date/time display
       - User controls
    
    2. Main Visualization Area
       - Health map visualization (interactive)
       - Risk map visualization
       - Vegetation index maps
       - Time series charts
    
    3. Control Panel
       - Field selection dropdown
       - Date range selector
       - Visualization type selector
       - Alert filtering options
    
    4. Alert Panel
       - Critical alerts (red)
       - Warning alerts (yellow)
       - Informational alerts (blue)
       - Alert details and recommendations
    
    5. Statistics Panel
       - Overall field health metrics
       - Risk distribution
       - Historical comparisons
       - Treatment recommendations
    
    6. Export Options
       - Report generation
       - Data export (CSV, GeoTIFF)
       - Image download
    """
    
    print("Dashboard Layout:")
    print(layout_description)
    
    return layout_description

# Example dashboard components
def dashboard_components_example():
    """
    Example of dashboard components structure
    """
    components = {
        'header': {
            'title': 'AI-Powered Spectral Health Mapping System',
            'subtitle': 'Real-time Crop Monitoring and Decision Support',
            'logo': 'path/to/logo.png'
        },
        'main_visualizations': {
            'health_map': {
                'type': 'heatmap',
                'colormap': 'RdYlGn_r',
                'interactive': True,
                'zoomable': True
            },
            'risk_map': {
                'type': 'heatmap',
                'colormap': 'Reds',
                'interactive': True
            },
            'veg_index': {
                'type': 'heatmap',
                'colormap': 'RdYlGn',
                'interactive': True
            }
        },
        'controls': {
            'field_selector': ['Field A', 'Field B', 'Field C'],
            'date_range': 'last_30_days',
            'visualization_type': ['Health Map', 'Risk Map', 'NDVI Map']
        },
        'alerts': {
            'critical': [],
            'warning': [],
            'info': []
        },
        'statistics': {
            'overall_health': 0.75,
            'risk_level': 'Moderate',
            'treatment_recommendations': ['Irrigation Adjustment', 'Fertilizer Application']
        }
    }
    
    return components

# Display example
dashboard_layout = create_dashboard_layout()
dashboard_components = dashboard_components_example()
print("Dashboard Components Example:")
for key, value in dashboard_components.items():
    print(f"  {key}: {type(value)}")
```

## Alert Visualization

### Alert Dashboard

Creating visual alerts for immediate attention:

```python
def visualize_alerts(alerts, field_map_shape):
    """
    Visualize alerts on a field map
    """
    # Create base map
    alert_map = np.zeros(field_map_shape)
    
    # Mark alert locations
    alert_colors = {'CRITICAL': 3, 'WARNING': 2, 'INFO': 1}
    
    plt.figure(figsize=(12, 10))
    
    # Plot base field (in light green)
    base_field = np.full(field_map_shape, 0.5)
    plt.imshow(base_field, cmap='Greens', alpha=0.3)
    
    # Plot alerts
    for alert in alerts:
        if 'coordinates' in alert:
            for coord in alert['coordinates'][:10]:  # Limit to first 10 for visibility
                y, x = coord
                if 0 <= y < field_map_shape[0] and 0 <= x < field_map_shape[1]:
                    alert_level = alert_colors.get(alert['level'], 1)
                    plt.scatter(x, y, c=alert_level, s=50, cmap='RdYlGn_r', 
                              vmin=1, vmax=3, edgecolors='black', linewidth=0.5)
    
    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightgreen', alpha=0.3, label='Field Area'),
        plt.scatter([], [], c=3, cmap='RdYlGn_r', vmin=1, vmax=3, 
                   label='Critical Alert', edgecolors='black', linewidth=0.5),
        plt.scatter([], [], c=2, cmap='RdYlGn_r', vmin=1, vmax=3, 
                   label='Warning Alert', edgecolors='black', linewidth=0.5),
        plt.scatter([], [], c=1, cmap='RdYlGn_r', vmin=1, vmax=3, 
                   label='Info Alert', edgecolors='black', linewidth=0.5)
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('Field Alert Map', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example usage
sample_alerts = [
    {
        'level': 'CRITICAL',
        'message': 'Disease outbreak detected',
        'coordinates': [(50, 60), (52, 62), (48, 58)],
        'recommended_action': 'Immediate treatment required'
    },
    {
        'level': 'WARNING',
        'message': 'Water stress detected',
        'coordinates': [(30, 40), (32, 42)],
        'recommended_action': 'Increase irrigation'
    }
]

visualize_alerts(sample_alerts, (128, 128))
```

## Decision Support Visualization

### Recommendation Dashboard

Presenting treatment recommendations visually:

```python
def visualize_recommendations(recommendations):
    """
    Visualize treatment recommendations
    """
    if not recommendations:
        print("No recommendations available.")
        return
    
    # Create urgency levels
    urgency_colors = {
        'immediate': '#DC143C',      # Red
        'within_24h': '#FF8C00',     # Orange
        'within_week': '#FFD700',    # Yellow
        'monitor': '#2E8B57'         # Green
    }
    
    # Create bar chart of recommendations by urgency
    urgencies = [rec.get('urgency', 'monitor') for rec in recommendations]
    urgency_counts = {}
    for urgency in urgencies:
        urgency_counts[urgency] = urgency_counts.get(urgency, 0) + 1
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(urgency_counts.keys(), urgency_counts.values(), 
                   color=[urgency_colors.get(k, '#808080') for k in urgency_counts.keys()])
    
    plt.title('Treatment Recommendations by Urgency', fontsize=16)
    plt.xlabel('Urgency Level')
    plt.ylabel('Number of Recommendations')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed recommendations
    print("\nDetailed Recommendations:")
    print("=" * 50)
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.get('problem_type', 'Unknown Problem')}")
        print(f"   Urgency: {rec.get('urgency', 'monitor')}")
        print(f"   Treatments: {', '.join(rec.get('treatments', []))}")
        print(f"   Effectiveness: {rec.get('expected_effectiveness', 0)*100:.0f}%")
        if 'estimated_cost' in rec:
            cost_info = rec['estimated_cost']
            print(f"   Estimated Cost: ${cost_info.get('total_cost_usd', 0):.2f}")

# Example usage
sample_recommendations = [
    {
        'problem_type': 'Fungal Disease',
        'urgency': 'immediate',
        'treatments': ['Fungicide Application', 'Remove Infected Plants'],
        'expected_effectiveness': 0.85,
        'estimated_cost': {'total_cost_usd': 250.00, 'cost_per_hectare': 25.00}
    },
    {
        'problem_type': 'Water Stress',
        'urgency': 'within_24h',
        'treatments': ['Irrigation Adjustment', 'Mulching'],
        'expected_effectiveness': 0.90,
        'estimated_cost': {'total_cost_usd': 120.00, 'cost_per_hectare': 12.00}
    },
    {
        'problem_type': 'Nutrient Deficiency',
        'urgency': 'within_week',
        'treatments': ['Fertilizer Application', 'Soil Amendment'],
        'expected_effectiveness': 0.80,
        'estimated_cost': {'total_cost_usd': 180.00, 'cost_per_hectare': 18.00}
    }
]

visualize_recommendations(sample_recommendations)
```

## Practical Exercises

### Exercise 1: Creating a Comprehensive Field Report

```python
def create_field_report(health_map, risk_map, ndvi_map, alerts, recommendations):
    """
    Create a comprehensive field report with multiple visualizations
    """
    fig = plt.figure(figsize=(20, 15))
    
    # Health Map
    ax1 = plt.subplot(2, 3, 1)
    im1 = ax1.imshow(health_map, cmap=ListedColormap(['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']), 
                     vmin=0, vmax=3)
    ax1.set_title('Health Classification Map', fontsize=14)
    ax1.axis('off')
    
    # Risk Map
    ax2 = plt.subplot(2, 3, 2)
    im2 = ax2.imshow(risk_map, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax2.set_title('Risk Assessment Map', fontsize=14)
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # NDVI Map
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(ndvi_map, cmap='RdYlGn', vmin=-1, vmax=1)
    ax3.set_title('NDVI Map', fontsize=14)
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Health Distribution
    ax4 = plt.subplot(2, 3, 4)
    health_counts = [np.sum(health_map == i) for i in range(4)]
    health_labels = ['Healthy', 'Mild Stress', 'Moderate Stress', 'Severe Stress']
    colors = ['#2E8B57', '#FFD700', '#FF8C00', '#DC143C']
    ax4.pie(health_counts, labels=health_labels, colors=colors, autopct='%1.1f%%')
    ax4.set_title('Health Distribution', fontsize=14)
    
    # Risk Distribution
    ax5 = plt.subplot(2, 3, 5)
    risk_categories = [
        np.sum(risk_map < 0.3),  # Low risk
        np.sum((risk_map >= 0.3) & (risk_map < 0.6)),  # Moderate risk
        np.sum(risk_map >= 0.6)  # High risk
    ]
    risk_labels = ['Low Risk', 'Moderate Risk', 'High Risk']
    risk_colors = ['#2E8B57', '#FFD700', '#DC143C']
    ax5.pie(risk_categories, labels=risk_labels, colors=risk_colors, autopct='%1.1f%%')
    ax5.set_title('Risk Distribution', fontsize=14)
    
    # Alert Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    alert_text = "Alert Summary:\n\n"
    for alert in alerts[:3]:  # Show first 3 alerts
        alert_text += f"• {alert['level']}: {alert['message'][:30]}...\n"
    if len(alerts) > 3:
        alert_text += f"... and {len(alerts) - 3} more alerts"
    ax6.text(0.1, 0.9, alert_text, transform=ax6.transAxes, fontsize=12, 
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax6.set_title('Alert Summary', fontsize=14)
    
    plt.suptitle('Comprehensive Field Health Report', fontsize=20)
    plt.tight_layout()
    plt.show()

# Example usage
# Generate sample data
sample_health = np.random.choice([0, 1, 2, 3], size=(100, 100), p=[0.6, 0.25, 0.1, 0.05])
sample_risk = np.random.beta(1.5, 2, (100, 100))
sample_ndvi = np.random.normal(0.6, 0.2, (100, 100))
sample_ndvi = np.clip(sample_ndvi, -1, 1)

sample_alerts = [
    {'level': 'CRITICAL', 'message': 'Disease outbreak detected in northern section'},
    {'level': 'WARNING', 'message': 'Water stress in eastern quadrant'},
    {'level': 'INFO', 'message': 'Normal growth patterns observed in southern area'}
]

sample_recommendations = [
    {'problem_type': 'Fungal Disease', 'urgency': 'immediate'},
    {'problem_type': 'Water Stress', 'urgency': 'within_24h'}
]

create_field_report(sample_health, sample_risk, sample_ndvi, sample_alerts, sample_recommendations)
```

### Exercise 2: Interactive Alert Map

```python
def create_interactive_alert_map(health_map, risk_map, alerts):
    """
    Create an interactive map showing alerts with details
    """
    # This is a conceptual example for a more advanced implementation
    print("Interactive Alert Map Concept:")
    print("=" * 40)
    print("Features would include:")
    print("1. Zoom and pan capabilities")
    print("2. Clickable alert markers")
    print("3. Hover-over information")
    print("4. Filter by alert type")
    print("5. Export functionality")
    print("6. Real-time updates")
    
    # Simple text-based representation
    print("\nAlert Locations:")
    for i, alert in enumerate(alerts, 1):
        print(f"{i}. {alert['level']} Alert: {alert['message']}")
        if 'coordinates' in alert:
            print(f"   Coordinates: {alert['coordinates'][:3]}{'...' if len(alert['coordinates']) > 3 else ''}")
        print(f"   Recommended Action: {alert['recommended_action']}")
        print()

# Example usage
sample_alerts_with_coords = [
    {
        'level': 'CRITICAL',
        'message': 'Severe disease outbreak detected',
        'coordinates': [(25, 30), (27, 32), (23, 28), (26, 31)],
        'recommended_action': 'Immediate fungicide application required'
    },
    {
        'level': 'WARNING',
        'message': 'Moderate water stress identified',
        'coordinates': [(70, 80), (72, 82), (68, 78)],
        'recommended_action': 'Adjust irrigation schedule within 24 hours'
    },
    {
        'level': 'INFO',
        'message': 'Normal growth patterns observed',
        'coordinates': [(50, 50), (52, 52), (48, 48)],
        'recommended_action': 'Continue regular monitoring'
    }
]

create_interactive_alert_map(sample_health, sample_risk, sample_alerts_with_coords)
```

## Discussion Questions

1. How do different visualization techniques help farmers make better decisions?
2. What are the advantages and disadvantages of static vs. interactive visualizations in agricultural applications?
3. How can visualization design be optimized for users with varying levels of technical expertise?
4. What additional visualization features would be valuable for precision agriculture applications?

## Additional Resources

- Dash documentation for Python web applications
- Matplotlib and Seaborn visualization libraries
- Plotly for interactive visualizations
- GIS software for agricultural mapping
- Research papers on agricultural decision support systems

## Summary

This chapter covered the visualization and decision support capabilities of the AI-Powered Spectral Health Mapping System. We explored various visualization techniques for health maps, risk maps, vegetation indices, and time series data. We also discussed interactive dashboard implementation and alert visualization methods. Effective visualization is crucial for translating complex analytical results into actionable insights for farmers and agricultural professionals. In the next chapter, we'll explore the practical implementation and deployment of the system.