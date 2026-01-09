"""
File handling utilities
"""
import os
import uuid
import shutil
from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class FileManager:
    """File management utilities"""
    
    @staticmethod
    def create_upload_directory() -> str:
        """Create upload directory if it doesn't exist"""
        upload_dir = Path(settings.UPLOAD_FOLDER)
        upload_dir.mkdir(parents=True, exist_ok=True)
        return str(upload_dir)
    
    @staticmethod
    def generate_unique_filename(original_filename: str) -> str:
        """Generate unique filename while preserving extension"""
        file_ext = Path(original_filename).suffix
        unique_name = f"{uuid.uuid4()}{file_ext}"
        return unique_name
    
    @staticmethod
    def validate_file_size(file_path: str) -> bool:
        """Validate file size against maximum allowed"""
        try:
            file_size = os.path.getsize(file_path)
            return file_size <= settings.MAX_FILE_SIZE
        except OSError:
            return False
    
    @staticmethod
    def validate_file_extension(filename: str) -> bool:
        """Validate file extension"""
        file_ext = Path(filename).suffix.lower()
        return file_ext in settings.ALLOWED_EXTENSIONS
    
    @staticmethod
    def safe_delete_file(file_path: str) -> bool:
        """Safely delete file"""
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file {file_path}: {str(e)}")
            return False
    
    @staticmethod
    def move_file(source: str, destination: str) -> bool:
        """Move file from source to destination"""
        try:
            # Create destination directory if needed
            dest_dir = Path(destination).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.move(source, destination)
            logger.info(f"Moved file from {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error moving file: {str(e)}")
            return False
    
    @staticmethod
    def copy_file(source: str, destination: str) -> bool:
        """Copy file from source to destination"""
        try:
            # Create destination directory if needed
            dest_dir = Path(destination).parent
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(source, destination)
            logger.info(f"Copied file from {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error copying file: {str(e)}")
            return False
    
    @staticmethod
    def get_file_info(file_path: str) -> Optional[dict]:
        """Get file information"""
        try:
            if not os.path.exists(file_path):
                return None
            
            stat = os.stat(file_path)
            file_info = {
                "path": file_path,
                "filename": os.path.basename(file_path),
                "size_bytes": stat.st_size,
                "size_mb": stat.st_size / (1024 * 1024),
                "created_at": stat.st_ctime,
                "modified_at": stat.st_mtime,
                "extension": Path(file_path).suffix.lower()
            }
            
            return file_info
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            return None
    
    @staticmethod
    def cleanup_old_files(directory: str, max_age_hours: int = 24) -> int:
        """Clean up old files in directory"""
        import time
        
        cleaned_count = 0
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    file_mtime = os.path.getmtime(file_path)
                    
                    if file_mtime < cutoff_time:
                        if FileManager.safe_delete_file(file_path):
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} old files from {directory}")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up directory {directory}: {str(e)}")
            return 0


class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def load_image(file_path: str) -> Optional[np.ndarray]:
        """Load image as numpy array"""
        try:
            image = Image.open(file_path)
            return np.array(image)
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {str(e)}")
            return None
    
    @staticmethod
    def save_image(image_array: np.ndarray, file_path: str) -> bool:
        """Save numpy array as image"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Convert to PIL Image and save
            image = Image.fromarray(image_array.astype(np.uint8))
            image.save(file_path)
            
            logger.info(f"Saved image to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {str(e)}")
            return False
    
    @staticmethod
    def resize_image(image_array: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Resize image array"""
        try:
            image = Image.fromarray(image_array.astype(np.uint8))
            resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
            return np.array(resized_image)
        except Exception as e:
            logger.error(f"Error resizing image: {str(e)}")
            return image_array
    
    @staticmethod
    def normalize_image(image_array: np.ndarray) -> np.ndarray:
        """Normalize image array to 0-1 range"""
        try:
            min_val = np.min(image_array)
            max_val = np.max(image_array)
            
            if max_val > min_val:
                normalized = (image_array - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(image_array)
            
            return normalized
        except Exception as e:
            logger.error(f"Error normalizing image: {str(e)}")
            return image_array


# Global file manager instance
file_manager = FileManager()
image_processor = ImageProcessor()