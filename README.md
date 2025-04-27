# 🐶 End-to-End Dog Breed Classification with TensorFlow Hub

This project builds a multi-class image classifier to identify the breed of a dog from a given image, using **TensorFlow 2.x** and **TensorFlow Hub**.

---

## 📚 Problem Statement

> Identify the breed of a dog given its image.

Imagine sitting at a café, spotting a cute dog, and instantly knowing its breed — this model aims to enable that!

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

---

## 🏗️ Project Structure

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

- Python
- TensorFlow 2.x
- TensorFlow Hub
- Pandas
- Matplotlib
- NumPy

---

## ⚡ Notes

- Data loading paths (e.g., `/content/drive/MyDrive/Dog Vision/labels.csv`) suggest the project was run in a Google Colab environment. So, change it accordingly.

---

## ✨ Future Improvements

- Fine-tune the model for higher accuracy.
- Experiment with different TensorFlow Hub models.
- Add real-time dog breed detection using webcam input.

---
