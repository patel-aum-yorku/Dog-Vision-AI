# 🐶 End-to-End Dog Breed Classification with TensorFlow Hub

This project builds a multi-class image classifier to identify the breed of a dog from a given image, using **TensorFlow 2.x** and **TensorFlow Hub**. The project now includes a **FastAPI backend service** for easy deployment and mobile app integration.

---

## 📚 Problem Statement

> Identify the breed of a dog given its image.

Imagine sitting at a café, spotting a cute dog, and instantly knowing its breed — this model aims to enable that!

---

## 🏗️ Project Structure

```
Dog-Vision-AI/
├── end_to_end_dog_vision.ipynb           # Main training notebook
├── 20250421-07101745219413-10k-images-mobilenetv2-Adam.keras  # Trained model
├── backend/                              # FastAPI backend service
│   ├── app.py                           # Main FastAPI application
│   ├── model_service.py                 # Model loading and prediction logic
│   ├── image_processor.py               # Image preprocessing functions
│   ├── requirements.txt                 # Python dependencies
│   ├── README.md                        # Backend setup instructions
│   ├── validate_setup.py                # Structure validation script
│   └── test_api.py                      # API testing script
├── README.md                            # This file
└── LICENSE                              # MIT License
```

---

## 🚀 Backend API Service

### Quick Start

1. **Navigate to backend directory:**
   ```bash
   cd backend
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the service:**
   ```bash
   python app.py
   ```

4. **Test the API:**
   - Health check: `http://localhost:8000/health`
   - API docs: `http://localhost:8000/docs`
   - Upload image for prediction via `/predict` endpoint

### API Endpoints

- **POST /predict**: Upload an image and get breed prediction
- **GET /health**: Check service and model status  
- **GET /**: Service information

### Example Usage

```bash
# Health check
curl http://localhost:8000/health

# Predict dog breed
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@dog_image.jpg"
```

**Response:**
```json
{
  "status": "success",
  "predicted_breed": "golden_retriever", 
  "confidence": 0.9234,
  "message": "Prediction successful"
}
```

For detailed backend documentation, see [backend/README.md](backend/README.md).

---

## 📊 Dataset

- **Source**: [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/c/dog-breed-identification/data)
- **Contents**:
  - `train.zip`: Training images with corresponding breed labels.
  - `test.zip`: Test images without labels (for evaluation).
  - `sample_submission.csv`: Example submission format.
  - `labels.csv`: Mapping of training images to dog breeds.
- **Details**:
  - 10,000+ labeled training images.
  - 10,000+ unlabeled test images.
  - 120 different dog breeds.

---

## 📈 Evaluation Metric

- Submission should predict a probability for each breed for each test image.
- Metric details can be found [here](https://www.kaggle.com/c/dog-breed-identification/overview).

---

## 🛠️ Features and Approach

- **Image Data**: Since images are unstructured data, deep learning and transfer learning are suitable approaches.
- **Transfer Learning**: Using pre-trained models from TensorFlow Hub.
- **Hardware Acceleration**: Ensures the use of a GPU if available for faster training and inference.
- **Production Ready**: FastAPI backend service for easy deployment and integration.

---

## 🏗️ Machine Learning Pipeline

1. **Setup**:
   - Import libraries (`tensorflow`, `tensorflow_hub`, `pandas`, `matplotlib`).
   - Verify GPU availability.
2. **Data Preparation**:
   - Load `labels.csv` for training data.
   - Analyze distribution of breeds.
   - Visualize number of images per breed.
3. **Model Building**:
   - Use TensorFlow Hub modules to leverage pre-trained feature extractors.
4. **Training and Evaluation**:
   - Train the model on labeled images.
   - Evaluate performance visually and numerically.
5. **Prediction**:
   - Generate predictions on the test set in the required submission format.

---

## 📦 Dependencies

### For Notebook:
- Python 3.8+
- TensorFlow 2.x
- TensorFlow Hub
- Pandas
- Matplotlib
- NumPy

### For Backend Service:
- FastAPI
- Uvicorn
- TensorFlow 2.16+
- Pillow
- Python-multipart
- NumPy

---

## ⚡ Notes

- Data loading paths (e.g., `/content/drive/MyDrive/Dog Vision/labels.csv`) suggest the project was run in a Google Colab environment. So, change it accordingly.
- The trained model file is included in the repository for immediate use with the backend service.

---

## ✨ Future Improvements

- Fine-tune the model for higher accuracy.
- Experiment with different TensorFlow Hub models.
- Add real-time dog breed detection using webcam input.
- Add user authentication to the API service.
- Implement batch prediction endpoints.
- Add model performance monitoring and logging.

---
