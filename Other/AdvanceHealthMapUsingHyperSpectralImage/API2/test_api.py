"""
Test server and validation script for the Spectral Health Mapping API
"""
import requests
import json
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np
from PIL import Image
import pandas as pd
import logging
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APITester:
    """Class to test the Spectral Health Mapping API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API tester
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
        self.api_prefix = "/api/v1"
        
    def test_health_endpoint(self) -> bool:
        """Test the health endpoint"""
        try:
            logger.info("Testing health endpoint...")
            response = requests.get(f"{self.base_url}/health", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health check passed: {data['status']}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check error: {str(e)}")
            return False
    
    def test_info_endpoint(self) -> bool:
        """Test the info endpoint"""
        try:
            logger.info("Testing info endpoint...")
            response = requests.get(f"{self.base_url}/info", timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"API Info: {data['api']['name']} v{data['api']['version']}")
                return True
            else:
                logger.error(f"Info endpoint failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Info endpoint error: {str(e)}")
            return False
    
    def test_model_api_endpoint(self) -> bool:
        """Test the model API endpoint with JSON input"""
        try:
            logger.info("Testing model API endpoint...")
            
            # Prepare test data
            test_data = {
                "field_id": "test_field_001",
                "user_id": "test_user_001",
                "hyperspectral_image_url": "http://example.com/test_image.tif",
                "sensor_data": {
                    "air_temperature": 25.5,
                    "humidity": 65.0,
                    "soil_moisture": 45.0,
                    "timestamp": "2025-10-04T10:00:00Z",
                    "wind_speed": 3.2,
                    "rainfall": 2.5,
                    "solar_radiation": 800.0,
                    "leaf_wetness": 35.0,
                    "co2_level": 420.0,
                    "ph_level": 6.8
                }
            }
            
            response = requests.post(
                f"{self.base_url}{self.api_prefix}/predict/model-api",
                json=test_data,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Model API test passed: {data['status']}")
                logger.info(f"Health score: {data.get('crop_health_status', {}).get('overall_health_score', 'N/A')}")
                return True
            else:
                logger.error(f"Model API test failed: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Model API test error: {str(e)}")
            return False
    
    def test_file_upload_endpoint(self) -> bool:
        """Test the file upload endpoint"""
        try:
            logger.info("Testing file upload endpoint...")
            
            # Create test files
            test_files = self._create_test_files()
            
            # Prepare form data
            files = {}
            data = {
                "field_id": "test_field_002",
                "user_id": "test_user_002"
            }
            
            # Add TIF file if created
            if test_files.get('tif_path'):
                files['tif_file'] = open(test_files['tif_path'], 'rb')
            
            # Add CSV file if created
            if test_files.get('csv_path'):
                files['csv_file'] = open(test_files['csv_path'], 'rb')
            
            try:
                response = requests.post(
                    f"{self.base_url}{self.api_prefix}/predict/upload-files",
                    data=data,
                    files=files,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"File upload test passed: {result['status']}")
                    logger.info(f"Health score: {result.get('crop_health_status', {}).get('overall_health_score', 'N/A')}")
                    return True
                else:
                    logger.error(f"File upload test failed: {response.status_code}")
                    logger.error(f"Response: {response.text}")
                    return False
                    
            finally:
                # Close files
                for file in files.values():
                    if hasattr(file, 'close'):
                        file.close()
                
                # Clean up test files
                self._cleanup_test_files(test_files)
                
        except Exception as e:
            logger.error(f"File upload test error: {str(e)}")
            return False
    
    def _create_test_files(self) -> Dict[str, str]:
        """Create test files for upload testing"""
        test_files = {}
        
        try:
            # Create test TIF image
            test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            image = Image.fromarray(test_image)
            
            tif_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
            image.save(tif_file.name, format='TIFF')
            test_files['tif_path'] = tif_file.name
            
            # Create test CSV sensor data
            sensor_data = {
                'air_temperature': [22.5, 23.1, 24.2],
                'humidity': [60.0, 62.5, 58.3],
                'soil_moisture': [40.0, 42.5, 38.7],
                'wind_speed': [2.5, 3.1, 2.8],
                'rainfall': [0.0, 1.2, 0.5],
                'solar_radiation': [750.0, 820.0, 680.0],
                'leaf_wetness': [30.0, 35.0, 28.0],
                'co2_level': [415.0, 418.0, 412.0],
                'ph_level': [6.5, 6.8, 6.3]
            }
            
            df = pd.DataFrame(sensor_data)
            csv_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            df.to_csv(csv_file.name, index=False)
            test_files['csv_path'] = csv_file.name
            
            logger.info("Created test files successfully")
            
        except Exception as e:
            logger.error(f"Error creating test files: {str(e)}")
        
        return test_files
    
    def _cleanup_test_files(self, test_files: Dict[str, str]):
        """Clean up test files"""
        for file_path in test_files.values():
            try:
                Path(file_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete test file {file_path}: {str(e)}")
    
    def run_all_tests(self) -> bool:
        """Run all API tests"""
        logger.info("Starting comprehensive API testing...")
        
        tests = [
            ("Health Endpoint", self.test_health_endpoint),
            ("Info Endpoint", self.test_info_endpoint),
            ("Model API Endpoint", self.test_model_api_endpoint),
            ("File Upload Endpoint", self.test_file_upload_endpoint)
        ]
        
        results = []
        
        for test_name, test_func in tests:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running: {test_name}")
            logger.info('='*50)
            
            try:
                result = test_func()
                results.append((test_name, result))
                
                if result:
                    logger.info(f"✅ {test_name} PASSED")
                else:
                    logger.error(f"❌ {test_name} FAILED")
                    
            except Exception as e:
                logger.error(f"❌ {test_name} ERROR: {str(e)}")
                results.append((test_name, False))
            
            time.sleep(1)  # Brief pause between tests
        
        # Summary
        logger.info(f"\n{'='*50}")
        logger.info("TEST SUMMARY")
        logger.info('='*50)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        return passed == total


def main():
    """Main function to run API tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Spectral Health Mapping API")
    parser.add_argument(
        "--url", 
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--test",
        choices=["health", "info", "model-api", "upload", "all"],
        default="all",
        help="Specific test to run (default: all)"
    )
    
    args = parser.parse_args()
    
    tester = APITester(base_url=args.url)
    
    # Run specific test or all tests
    if args.test == "health":
        success = tester.test_health_endpoint()
    elif args.test == "info":
        success = tester.test_info_endpoint()
    elif args.test == "model-api":
        success = tester.test_model_api_endpoint()
    elif args.test == "upload":
        success = tester.test_file_upload_endpoint()
    else:
        success = tester.run_all_tests()
    
    if success:
        logger.info("\n🎉 All tests completed successfully!")
        exit(0)
    else:
        logger.error("\n💥 Some tests failed!")
        exit(1)


if __name__ == "__main__":
    main()