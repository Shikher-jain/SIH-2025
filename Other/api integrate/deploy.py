#!/usr/bin/env python3
"""
Deployment script for the Multi-Modal CNN API
"""
import uvicorn
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def deploy_api(host="0.0.0.0", port=8000, workers=1):
    """
    Deploy the API for production use
    
    Args:
        host: Host to bind to (0.0.0.0 for all interfaces)
        port: Port to run on
        workers: Number of worker processes
    """


    
    print(f"🚀 Deploying Multi-Modal CNN API...")
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"👥 Workers: {workers}")
    print(f"📊 API Docs: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/docs")
    print(f"🔄 API Status: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/predict/status")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=False,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deploy Multi-Modal CNN API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    args = parser.parse_args()
    
    deploy_api(args.host, args.port, args.workers)