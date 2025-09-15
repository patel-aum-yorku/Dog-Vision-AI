"""
FastAPI backend service for Dog Vision AI.

This service provides endpoints for dog breed prediction using a trained TensorFlow model.
"""

import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from model_service import ModelService
from image_processor import (
    process_image_from_bytes, 
    prepare_image_for_prediction, 
    validate_image_format
)


# Initialize FastAPI app
app = FastAPI(
    title="Dog Vision AI",
    description="A FastAPI service for dog breed classification using deep learning",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Global model service instance
model_service = None

# Model file path (relative to the backend directory)
MODEL_PATH = "../20250421-07101745219413-10k-images-mobilenetv2-Adam.keras"

# Supported image formats
SUPPORTED_FORMATS = {"image/jpeg", "image/jpg", "image/png"}

# Maximum file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Rate limiting configuration
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "10"))
RATE_LIMIT_PERIOD = os.getenv("RATE_LIMIT_PERIOD", "minute")
RATE_LIMIT_STRING = f"{RATE_LIMIT_REQUESTS}/{RATE_LIMIT_PERIOD}"


@app.on_event("startup")
async def startup_event():
    """Initialize the model service on application startup."""
    global model_service
    
    # Get absolute path to model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_PATH)
    
    # Initialize model service
    model_service = ModelService(model_path)
    
    # Load the model
    success = model_service.load_model()
    if not success:
        print("Warning: Failed to load model on startup")


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify service status and model availability.
    
    Returns:
        dict: Service health status
    """
    if model_service is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": "Model service not initialized",
                "model_loaded": False
            }
        )
    
    health_status = model_service.get_health_status()
    
    return {
        "status": "success" if health_status["model_loaded"] else "warning",
        "message": "Service is running",
        **health_status
    }


@app.post("/predict")
@limiter.limit(RATE_LIMIT_STRING)
async def predict_dog_breed(request: Request, file: UploadFile = File(...)):
    """
    Predict dog breed from uploaded image.
    
    Rate limited to prevent abuse (default: 10 requests per minute per IP address).
    Rate limits can be configured via RATE_LIMIT_REQUESTS and RATE_LIMIT_PERIOD environment variables.
    
    Args:
        request: FastAPI request object (for rate limiting)
        file: Uploaded image file (JPEG or PNG)
        
    Returns:
        dict: Prediction result with breed name and confidence score
        
    Raises:
        HTTPException: 429 if rate limit is exceeded
        HTTPException: 400 for invalid file format or size
        HTTPException: 503 if model service is unavailable
        HTTPException: 500 for internal server errors
    """
    # Validate model service
    if model_service is None or not model_service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model service not available. Please check service health."
        )
    
    # Validate file format
    if file.content_type not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size allowed: {MAX_FILE_SIZE / (1024*1024):.1f}MB"
            )
        
        # Validate image format
        if not validate_image_format(file_content):
            raise HTTPException(
                status_code=400,
                detail="Invalid image file or corrupted image"
            )
        
        # Process image
        processed_image = process_image_from_bytes(file_content)
        
        # Prepare for prediction (add batch dimension)
        image_batch = prepare_image_for_prediction(processed_image)
        
        # Make prediction
        predicted_breed, confidence = model_service.predict(image_batch)
        
        return {
            "status": "success",
            "predicted_breed": predicted_breed,
            "confidence": round(confidence, 4),
            "message": "Prediction successful"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle any other errors
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


@app.get("/")
async def root():
    """
    Root endpoint with basic service information.
    
    Returns:
        dict: Basic service information
    """
    return {
        "message": "Dog Vision AI Backend Service",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )