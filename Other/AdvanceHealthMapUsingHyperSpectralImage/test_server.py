#!/usr/bin/env python3
"""
Simple test script to verify the FastAPI application loads correctly
"""

try:
    print("Testing imports...")
    
    print("1. Testing FastAPI import...")
    from fastapi import FastAPI
    print("✓ FastAPI imported successfully")
    
    print("2. Testing TensorFlow import...")
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    
    print("3. Testing our application imports...")
    from app.config import settings
    print("✓ Config imported successfully")
    
    print("4. Testing model module...")
    from app.models.cnn_model import CNNMultiModalModel
    print("✓ Model class imported successfully")
    
    print("5. Creating model instance...")
    model = CNNMultiModalModel()
    print(f"✓ Model instance created, ready: {model.is_model_ready()}")
    
    print("6. Testing full app import...")
    from app.main import app
    print("✓ Full app imported successfully")
    
    print("\n🎉 All tests passed! The application should work correctly.")
    print(f"Model path: {settings.MODEL_PATH}")
    print(f"Model file exists: {__import__('os').path.exists(settings.MODEL_PATH)}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()