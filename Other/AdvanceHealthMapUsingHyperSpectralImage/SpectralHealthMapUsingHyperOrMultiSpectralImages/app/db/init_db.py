"""
Database initialization utilities
"""
import logging
from sqlalchemy.orm import Session
from app.db.base import Base
from app.db.session import engine, SessionLocal
from app.models.user import User
from app.models.spectral import Field, SpectralAnalysis, ModelMetrics, AnalysisHistory
from app.core.security import SecurityManager

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Initialize database with tables"""
    try:
        # Import all models to ensure they are registered with Base
        import app.models.user
        import app.models.spectral
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise


def create_initial_data() -> None:
    """Create initial data (superuser, etc.)"""
    db = SessionLocal()
    try:
        # Check if superuser exists
        superuser = db.query(User).filter(User.is_superuser == True).first()
        
        if not superuser:
            # Create default superuser
            superuser = User(
                username="admin",
                email="admin@spectralhealth.com",
                full_name="System Administrator",
                hashed_password=SecurityManager.get_password_hash("admin123"),
                is_active=True,
                is_superuser=True,
                roles=["admin", "user", "researcher"]
            )
            db.add(superuser)
            db.commit()
            logger.info("Created default superuser: admin")
        
        # Create default model metrics
        existing_metrics = db.query(ModelMetrics).first()
        if not existing_metrics:
            default_metrics = [
                ModelMetrics(
                    model_name="cnn_disease_classifier",
                    model_version="1.0.0",
                    accuracy=0.92,
                    precision=0.89,
                    recall=0.94,
                    f1_score=0.91
                ),
                ModelMetrics(
                    model_name="lstm_progression_predictor",
                    model_version="1.0.0",
                    accuracy=0.88,
                    precision=0.85,
                    recall=0.90,
                    f1_score=0.87
                ),
                ModelMetrics(
                    model_name="autoencoder_anomaly_detector",
                    model_version="1.0.0",
                    accuracy=0.91,
                    precision=0.88,
                    recall=0.93,
                    f1_score=0.90
                )
            ]
            
            for metric in default_metrics:
                db.add(metric)
            
            db.commit()
            logger.info("Created default model metrics")
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating initial data: {e}")
        raise
    finally:
        db.close()


def reset_database() -> None:
    """Reset database (drop and recreate all tables)"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped")
        
        init_db()
        create_initial_data()
        logger.info("Database reset completed")
        
    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        raise


if __name__ == "__main__":
    # Initialize database when run directly
    logging.basicConfig(level=logging.INFO)
    init_db()
    create_initial_data()
    print("Database initialization completed")