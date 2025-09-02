# Dog Vision AI Backend Service

A FastAPI-based backend service for dog breed classification using a trained TensorFlow model. This service provides REST API endpoints to receive images from mobile phones and return dog breed predictions.

## Features

- **FastAPI Framework**: High-performance, easy-to-use web framework with automatic API documentation
- **Dog Breed Classification**: Predicts dog breeds from uploaded images using a trained MobileNetV2 model
- **Image Processing**: Handles JPEG and PNG image formats with preprocessing pipeline
- **Error Handling**: Comprehensive error handling for invalid images, file size limits, and model failures
- **Health Monitoring**: Health check endpoint to verify service and model status

## API Endpoints

### 1. POST /predict
Main prediction endpoint that accepts image uploads and returns breed predictions.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (JPEG or PNG, max 10MB)

**Response:**
```json
{
  "status": "success",
  "predicted_breed": "golden_retriever",
  "confidence": 0.9234,
  "message": "Prediction successful"
}
```

### 2. GET /health
Health check endpoint to verify service status and model availability.

**Response:**
```json
{
  "status": "success",
  "message": "Service is running",
  "model_loaded": true,
  "model_path": "../20250421-07101745219413-10k-images-mobilenetv2-Adam.keras",
  "num_classes": 120
}
```

### 3. GET /
Root endpoint with basic service information and available endpoints.

## Setup and Installation

### Prerequisites
- Python 3.8 or higher
- The trained Keras model file (`20250421-07101745219413-10k-images-mobilenetv2-Adam.keras`) in the parent directory

### Installation Steps

1. **Clone the repository and navigate to the backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify model file location:**
   Make sure the model file `20250421-07101745219413-10k-images-mobilenetv2-Adam.keras` is in the parent directory relative to the backend folder.

4. **Run the service:**
   ```bash
   python app.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Access the service:**
   - API Base URL: `http://localhost:8000`
   - Interactive API Documentation: `http://localhost:8000/docs`
   - Alternative API Documentation: `http://localhost:8000/redoc`

## Usage Examples

### Using curl
```bash
# Health check
curl http://localhost:8000/health

# Predict dog breed
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@dog_image.jpg"
```

### Using Python requests
```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Predict dog breed
with open("dog_image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

## Image Requirements

- **Supported Formats**: JPEG, PNG
- **Maximum File Size**: 10MB
- **Recommended**: Clear images of dogs, preferably showing the full dog or at least the head/face area

## Model Information

- **Architecture**: MobileNetV2 with transfer learning
- **Input Size**: 224x224 pixels, RGB
- **Number of Classes**: 120 dog breeds
- **Preprocessing**: Images are automatically resized to 224x224, converted to RGB, and normalized

## Error Handling

The service handles various error conditions:

- **Invalid file format**: Returns 400 error for unsupported image formats
- **File too large**: Returns 400 error for files exceeding 10MB
- **Corrupted images**: Returns 400 error for invalid/corrupted image files
- **Model not loaded**: Returns 503 error if the model is not available
- **Processing errors**: Returns 500 error for unexpected processing failures

## Development

### File Structure
```
backend/
├── app.py                    # Main FastAPI application
├── model_service.py          # Model loading and prediction logic
├── image_processor.py        # Image preprocessing functions
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

### Adding New Features
1. **New endpoints**: Add them to `app.py`
2. **Model changes**: Modify `model_service.py`
3. **Image processing**: Update `image_processor.py`

## Deployment

For production deployment:

1. **Use a production ASGI server**:
   ```bash
   pip install gunicorn
   gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Environment variables**: Consider using environment variables for configuration
3. **Monitoring**: Add logging and monitoring for production use
4. **Security**: Implement authentication if needed
5. **Scale**: Use load balancers and multiple instances for high traffic

## Troubleshooting

### Common Issues

1. **Model not found**:
   - Verify the model file path in `app.py`
   - Ensure the model file exists in the specified location

2. **Memory issues**:
   - Large images may cause memory issues
   - Consider implementing image resizing before processing

3. **Slow predictions**:
   - Consider using GPU acceleration if available
   - Implement model caching for better performance

### Logs
The service provides console logging for debugging. Check the console output for error messages and status information.

## License

This project is licensed under the MIT License - see the LICENSE file for details.