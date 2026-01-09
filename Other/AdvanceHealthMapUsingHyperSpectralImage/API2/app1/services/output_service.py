"""
Output service for generating health maps, reports, and data exports
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import base64
from io import BytesIO

from ..config import settings
from ..core.exceptions import ImageProcessingException

logger = logging.getLogger(__name__)


class HealthMapGenerator:
    """Generate visual health maps from prediction results"""
    
    @staticmethod
    def generate_health_map(
        original_image: np.ndarray,
        prediction_results: Dict[str, Any],
        field_id: str
    ) -> str:
        """
        Generate a color-coded health map overlay
        
        Args:
            original_image: Original hyperspectral image
            prediction_results: Model prediction results
            field_id: Field identifier
            
        Returns:
            Path to generated health map image
        """
        try:
            # Create figure and subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Crop Health Analysis - Field {field_id}', fontsize=16, fontweight='bold')
            
            # Original image
            axes[0, 0].imshow(original_image)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Health map overlay
            health_score = prediction_results.get('health_score', 50)
            health_category = prediction_results.get('health_category', 'Fair')
            
            # Create health map based on score
            health_map = HealthMapGenerator._create_health_overlay(original_image, health_score)
            axes[0, 1].imshow(health_map)
            axes[0, 1].set_title(f'Health Map - {health_category} ({health_score:.1f}%)')
            axes[0, 1].axis('off')
            
            # NDVI visualization
            ndvi_value = prediction_results.get('vegetation_indices', {}).get('ndvi', 0.5)
            ndvi_map = HealthMapGenerator._create_ndvi_map(original_image, ndvi_value)
            axes[1, 0].imshow(ndvi_map)
            axes[1, 0].set_title(f'NDVI Map (Value: {ndvi_value:.3f})')
            axes[1, 0].axis('off')
            
            # Stress indicators visualization
            stress_indicators = prediction_results.get('stress_indicators', [])
            HealthMapGenerator._plot_stress_analysis(axes[1, 1], prediction_results)
            
            # Add color bars and legends
            HealthMapGenerator._add_health_colorbar(fig, health_score)
            
            # Save the figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_map_{field_id}_{timestamp}.png"
            output_path = settings.get_output_path(filename)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated health map: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating health map: {str(e)}")
            raise ImageProcessingException(f"Failed to generate health map: {str(e)}")
    
    @staticmethod
    def _create_health_overlay(image: np.ndarray, health_score: float) -> np.ndarray:
        """Create color-coded health overlay"""
        # Define health colors
        colors = {
            'critical': [0.8, 0.2, 0.2],    # Red
            'poor': [1.0, 0.4, 0.2],        # Orange
            'fair': [1.0, 1.0, 0.2],        # Yellow
            'good': [0.6, 1.0, 0.2],        # Light Green
            'excellent': [0.2, 0.8, 0.2]    # Green
        }
        
        # Determine color based on health score
        if health_score < 20:
            color = colors['critical']
        elif health_score < 40:
            color = colors['poor']
        elif health_score < 60:
            color = colors['fair']
        elif health_score < 80:
            color = colors['good']
        else:
            color = colors['excellent']
        
        # Create overlay
        overlay = np.ones_like(image) * np.array(color)
        
        # Blend with original image
        alpha = 0.4
        health_map = (1 - alpha) * image + alpha * overlay
        
        return np.clip(health_map, 0, 1)
    
    @staticmethod
    def _create_ndvi_map(image: np.ndarray, ndvi_value: float) -> np.ndarray:
        """Create NDVI visualization map"""
        # Create NDVI colormap
        cmap = plt.cm.RdYlGn
        
        # Generate NDVI-like pattern based on image intensity
        gray_image = np.mean(image, axis=2)
        ndvi_pattern = gray_image * ndvi_value
        
        # Apply colormap
        ndvi_colored = cmap(ndvi_pattern)[:, :, :3]
        
        return ndvi_colored
    
    @staticmethod
    def _plot_stress_analysis(ax, prediction_results: Dict[str, Any]):
        """Plot stress indicators and risk analysis"""
        # Get data
        disease_risk = prediction_results.get('disease_risk', 0)
        pest_risk = prediction_results.get('pest_risk', 0)
        health_score = prediction_results.get('health_score', 50)
        
        # Create bar plot
        categories = ['Health Score', 'Disease Risk', 'Pest Risk']
        values = [health_score, disease_risk, pest_risk]
        colors = ['green', 'red', 'orange']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Score/Risk (%)')
        ax.set_title('Health & Risk Analysis')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    @staticmethod
    def _add_health_colorbar(fig, health_score: float):
        """Add color bar legend for health scores"""
        # Create colorbar
        cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        
        # Define health color map
        colors = ['#CC3333', '#FF6633', '#FFFF33', '#99FF33', '#33CC33']
        n_bins = 100
        cmap = mcolors.LinearSegmentedColormap.from_list('health', colors, N=n_bins)
        
        # Create colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
        sm.set_array([])
        cbar = plt.colorbar(sm, cax=cax)
        cbar.set_label('Health Score (%)', rotation=270, labelpad=20)
        
        # Mark current health score
        cbar.ax.axhline(health_score, color='black', linewidth=2, alpha=0.8)


class ReportGenerator:
    """Generate comprehensive analysis reports"""
    
    @staticmethod
    def generate_analysis_report(
        prediction_results: Dict[str, Any],
        field_info: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive analysis report
        
        Args:
            prediction_results: Model prediction results
            field_info: Field information
            metadata: Processing metadata
            
        Returns:
            Path to generated report file
        """
        try:
            # Extract data
            health_score = prediction_results.get('health_score', 0)
            health_category = prediction_results.get('health_category', 'Unknown')
            indices = prediction_results.get('vegetation_indices', {})
            stress_indicators = prediction_results.get('stress_indicators', [])
            recommendations = prediction_results.get('recommendations', [])
            
            # Generate report content
            report_content = ReportGenerator._create_report_content(
                prediction_results, field_info, metadata
            )
            
            # Save report as HTML
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_report_{field_info.get('field_id', 'unknown')}_{timestamp}.html"
            output_path = settings.get_output_path(filename)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            logger.info(f"Generated analysis report: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise ImageProcessingException(f"Failed to generate report: {str(e)}")
    
    @staticmethod
    def _create_report_content(
        prediction_results: Dict[str, Any],
        field_info: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> str:
        """Create HTML report content"""
        
        # Extract data
        health_score = prediction_results.get('health_score', 0)
        health_category = prediction_results.get('health_category', 'Unknown')
        indices = prediction_results.get('vegetation_indices', {})
        stress_indicators = prediction_results.get('stress_indicators', [])
        recommendations = prediction_results.get('recommendations', [])
        disease_risk = prediction_results.get('disease_risk', 0)
        pest_risk = prediction_results.get('pest_risk', 0)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crop Health Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2E8B57; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .health-score {{ font-size: 24px; font-weight: bold; text-align: center; margin: 20px 0; }}
                .excellent {{ color: #228B22; }}
                .good {{ color: #9ACD32; }}
                .fair {{ color: #FFD700; }}
                .poor {{ color: #FF6347; }}
                .critical {{ color: #DC143C; }}
                .indices-table {{ width: 100%; border-collapse: collapse; }}
                .indices-table th, .indices-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .indices-table th {{ background-color: #f2f2f2; }}
                ul {{ padding-left: 20px; }}
                .risk-box {{ display: inline-block; margin: 10px; padding: 15px; border-radius: 5px; }}
                .risk-low {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
                .risk-medium {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .risk-high {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Crop Health Analysis Report</h1>
                <p>Field ID: {field_info.get('field_id', 'N/A')} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Overall Health Assessment</h2>
                <div class="health-score {health_category.lower()}">
                    Health Category: {health_category}<br>
                    Health Score: {health_score:.1f}%
                </div>
            </div>
            
            <div class="section">
                <h2>Vegetation Indices</h2>
                <table class="indices-table">
                    <tr><th>Index</th><th>Value</th><th>Description</th></tr>
                    <tr><td>NDVI</td><td>{indices.get('ndvi', 0):.3f}</td><td>Normalized Difference Vegetation Index</td></tr>
                    <tr><td>EVI</td><td>{indices.get('evi', 0):.3f}</td><td>Enhanced Vegetation Index</td></tr>
                    <tr><td>SAVI</td><td>{indices.get('savi', 0):.3f}</td><td>Soil Adjusted Vegetation Index</td></tr>
                    <tr><td>NDWI</td><td>{indices.get('ndwi', 0):.3f}</td><td>Normalized Difference Water Index</td></tr>
                    <tr><td>PRI</td><td>{indices.get('pri', 0):.3f}</td><td>Photochemical Reflectance Index</td></tr>
                    <tr><td>Chlorophyll</td><td>{indices.get('chlorophyll_content', 0):.1f} mg/g</td><td>Chlorophyll Content</td></tr>
                    <tr><td>LAI</td><td>{indices.get('leaf_area_index', 0):.2f}</td><td>Leaf Area Index</td></tr>
                    <tr><td>Water Stress</td><td>{indices.get('water_stress_index', 0):.1f}%</td><td>Water Stress Index</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Risk Assessment</h2>
                <div class="risk-box {'risk-low' if disease_risk < 30 else 'risk-medium' if disease_risk < 70 else 'risk-high'}">
                    <strong>Disease Risk:</strong> {disease_risk:.1f}%
                </div>
                <div class="risk-box {'risk-low' if pest_risk < 30 else 'risk-medium' if pest_risk < 70 else 'risk-high'}">
                    <strong>Pest Risk:</strong> {pest_risk:.1f}%
                </div>
            </div>
            
            <div class="section">
                <h2>Stress Indicators</h2>
                {"<ul>" + "".join([f"<li>{indicator}</li>" for indicator in stress_indicators]) + "</ul>" if stress_indicators else "<p>No significant stress indicators detected.</p>"}
            </div>
            
            <div class="section">
                <h2>Management Recommendations</h2>
                {"<ul>" + "".join([f"<li>{rec}</li>" for rec in recommendations]) + "</ul>" if recommendations else "<p>No specific recommendations at this time.</p>"}
            </div>
            
            <div class="section">
                <h2>Processing Information</h2>
                <p><strong>Processing Time:</strong> {metadata.get('processing_time', 'N/A')}</p>
                <p><strong>Model Version:</strong> {metadata.get('model_version', 'N/A')}</p>
                <p><strong>Processing Duration:</strong> {metadata.get('processing_duration', 0):.2f} seconds</p>
                <p><strong>Image Resolution:</strong> {metadata.get('image_resolution', 'N/A')}</p>
                <p><strong>Bands Analyzed:</strong> {metadata.get('bands_analyzed', 'N/A')}</p>
            </div>
            
            <div class="section">
                <p><em>This report was generated by the Spectral Health Mapping API v{settings.APP_VERSION}. 
                For questions or support, please contact your agricultural advisor.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html_content


class DataExporter:
    """Export analysis data in various formats"""
    
    @staticmethod
    def export_indices_data(
        prediction_results: Dict[str, Any],
        field_info: Dict[str, str],
        format_type: str = 'json'
    ) -> str:
        """
        Export vegetation indices and analysis data
        
        Args:
            prediction_results: Model prediction results
            field_info: Field information
            format_type: Export format ('json' or 'csv')
            
        Returns:
            Path to exported data file
        """
        try:
            # Prepare data
            export_data = {
                'field_id': field_info.get('field_id', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'health_assessment': {
                    'health_score': prediction_results.get('health_score', 0),
                    'health_category': prediction_results.get('health_category', 'Unknown'),
                    'disease_risk': prediction_results.get('disease_risk', 0),
                    'pest_risk': prediction_results.get('pest_risk', 0)
                },
                'vegetation_indices': prediction_results.get('vegetation_indices', {}),
                'stress_indicators': prediction_results.get('stress_indicators', []),
                'recommendations': prediction_results.get('recommendations', [])
            }
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"indices_data_{field_info.get('field_id', 'unknown')}_{timestamp}.{format_type}"
            output_path = settings.get_output_path(filename)
            
            # Export based on format
            if format_type.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            elif format_type.lower() == 'csv':
                # Flatten data for CSV
                flat_data = DataExporter._flatten_data_for_csv(export_data)
                df = pd.DataFrame([flat_data])
                df.to_csv(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format_type}")
            
            logger.info(f"Exported indices data to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            raise ImageProcessingException(f"Failed to export data: {str(e)}")
    
    @staticmethod
    def _flatten_data_for_csv(data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested data structure for CSV export"""
        flat_data = {
            'field_id': data['field_id'],
            'timestamp': data['timestamp'],
            'health_score': data['health_assessment']['health_score'],
            'health_category': data['health_assessment']['health_category'],
            'disease_risk': data['health_assessment']['disease_risk'],
            'pest_risk': data['health_assessment']['pest_risk']
        }
        
        # Add vegetation indices
        for key, value in data['vegetation_indices'].items():
            flat_data[f'vi_{key}'] = value
        
        # Add stress indicators as comma-separated string
        flat_data['stress_indicators'] = '; '.join(data['stress_indicators'])
        
        # Add recommendations as comma-separated string
        flat_data['recommendations'] = '; '.join(data['recommendations'])
        
        return flat_data


class OutputService:
    """Main service for generating all types of outputs"""
    
    def __init__(self):
        self.health_map_generator = HealthMapGenerator()
        self.report_generator = ReportGenerator()
        self.data_exporter = DataExporter()
    
    def generate_all_outputs(
        self,
        original_image: np.ndarray,
        prediction_results: Dict[str, Any],
        field_info: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Generate all output files (health map, report, data exports)
        
        Args:
            original_image: Original input image
            prediction_results: Model prediction results
            field_info: Field information
            metadata: Processing metadata
            
        Returns:
            Dictionary with paths to generated files
        """
        try:
            outputs = {}
            
            # Generate health map
            health_map_path = self.health_map_generator.generate_health_map(
                original_image, prediction_results, field_info.get('field_id', 'unknown')
            )
            outputs['health_map'] = health_map_path
            
            # Generate analysis report
            report_path = self.report_generator.generate_analysis_report(
                prediction_results, field_info, metadata
            )
            outputs['report'] = report_path
            
            # Export indices data (JSON)
            json_path = self.data_exporter.export_indices_data(
                prediction_results, field_info, 'json'
            )
            outputs['indices_json'] = json_path
            
            # Export indices data (CSV)
            csv_path = self.data_exporter.export_indices_data(
                prediction_results, field_info, 'csv'
            )
            outputs['indices_csv'] = csv_path
            
            logger.info(f"Generated all outputs for field {field_info.get('field_id', 'unknown')}")
            return outputs
            
        except Exception as e:
            logger.error(f"Error generating outputs: {str(e)}")
            raise ImageProcessingException(f"Failed to generate outputs: {str(e)}")