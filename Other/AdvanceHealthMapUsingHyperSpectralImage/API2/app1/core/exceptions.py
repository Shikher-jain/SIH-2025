"""
Custom exceptions for the Spectral Health Mapping API
"""


class SpectralAPIException(Exception):
    """Base exception for Spectral API errors"""
    def __init__(self, message: str, error_code: str = "SPECTRAL_API_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ModelNotLoadedException(SpectralAPIException):
    """Exception raised when ML model is not loaded"""
    def __init__(self, message: str = "ML model is not loaded"):
        super().__init__(message, "MODEL_NOT_LOADED")


class InvalidFileFormatException(SpectralAPIException):
    """Exception raised for invalid file formats"""
    def __init__(self, message: str = "Invalid file format"):
        super().__init__(message, "INVALID_FILE_FORMAT")


class FileTooLargeException(SpectralAPIException):
    """Exception raised when uploaded file is too large"""
    def __init__(self, message: str = "File size exceeds maximum allowed size"):
        super().__init__(message, "FILE_TOO_LARGE")


class ImageProcessingException(SpectralAPIException):
    """Exception raised during image processing"""
    def __init__(self, message: str = "Error processing image"):
        super().__init__(message, "IMAGE_PROCESSING_ERROR")


class ModelPredictionException(SpectralAPIException):
    """Exception raised during model prediction"""
    def __init__(self, message: str = "Error during model prediction"):
        super().__init__(message, "PREDICTION_ERROR")


class DataValidationException(SpectralAPIException):
    """Exception raised for data validation errors"""
    def __init__(self, message: str = "Data validation failed"):
        super().__init__(message, "DATA_VALIDATION_ERROR")


class ResourceNotFoundException(SpectralAPIException):
    """Exception raised when requested resource is not found"""
    def __init__(self, message: str = "Requested resource not found"):
        super().__init__(message, "RESOURCE_NOT_FOUND")