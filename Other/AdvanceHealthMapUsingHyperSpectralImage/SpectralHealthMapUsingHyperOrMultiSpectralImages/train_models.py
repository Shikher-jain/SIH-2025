# train_models.py
#!/usr/bin/env python3
"""
Comprehensive Model Training Script for AI-Powered Spectral Health Mapping System
"""

import argparse
from main import SpectralHealthSystem, ModelTrainingSystem

def main():
    parser = argparse.ArgumentParser(description='Train AI models for spectral health mapping')
    parser.add_argument('--models', nargs='+', 
                       choices=['cnn', 'lstm', 'rf', 'autoencoder', 'unet', 'all'],
                       default=['all'], help='Models to train')
    parser.add_argument('--data', default='data/sample', help='Training data directory')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU acceleration')
    
    args = parser.parse_args()
    
    # Initialize training system
    trainer = ModelTrainingSystem(args.config)
    
    if 'all' in args.models:
        # Train all models
        models = trainer.train_all_models(args.data)
    else:
        # Train specific models
        models = {}
        if 'cnn' in args.models:
            models['cnn'] = trainer.train_cnn_model(trainer.load_training_data(args.data))
        if 'lstm' in args.models:
            models['lstm'] = trainer.train_lstm_model(trainer.load_training_data(args.data))
        if 'rf' in args.models:
            models['rf'] = trainer.train_random_forest(trainer.load_training_data(args.data))
        if 'autoencoder' in args.models:
            models['autoencoder'] = trainer.train_autoencoder(trainer.load_training_data(args.data))
        if 'unet' in args.models:
            models['unet'] = trainer.train_unet_model(trainer.load_training_data(args.data))
    
    print("🎉 Model training completed successfully!")
    print(f"📁 Models saved in: models/saved/")
    
    # Test the system with trained models
    system = SpectralHealthSystem(args.config)
    system.load_ai_models()
    print("✅ System ready with trained models!")

if __name__ == "__main__":
    main()