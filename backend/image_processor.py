"""
Image preprocessing module for Dog Vision AI backend.

This module contains functions to preprocess images for model inference,
replicating the preprocessing pipeline from the original notebook.
"""

import tensorflow as tf
from PIL import Image
import numpy as np
import io


# Image size constant (from notebook)
IMG_SIZE = 224


def process_image_from_bytes(image_bytes):
    """
    Process an image from bytes (uploaded file) into a tensor ready for model inference.
    
    This function replicates the process_image function from the notebook:
    1. Decode image from bytes
    2. Convert to RGB (3 channels)  
    3. Normalize pixel values (0-255 to 0-1)
    4. Resize to (224, 224)
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        
    Returns:
        tf.Tensor: Preprocessed image tensor of shape (224, 224, 3)
    """
    # Use PIL to open and convert image to RGB
    image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL image to numpy array
    image_array = np.array(image)
    
    # Convert to TensorFlow tensor
    image_tensor = tf.constant(image_array)
    
    # Convert to float32 and normalize (0-255 to 0-1)
    image_tensor = tf.image.convert_image_dtype(image_tensor, tf.float32)
    
    # Resize to the required size (224, 224)
    image_tensor = tf.image.resize(image_tensor, size=[IMG_SIZE, IMG_SIZE])
    
    return image_tensor


def process_image_from_file_path(image_path):
    """
    Process an image from file path into a tensor ready for model inference.
    
    This is the original process_image function from the notebook.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tf.Tensor: Preprocessed image tensor of shape (224, 224, 3)
    """
    # Read image file
    image = tf.io.read_file(image_path)
    
    # Decode JPEG image (assumes JPEG format)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Convert to float32 and normalize (0-255 to 0-1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize to the required size (224, 224)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    
    return image


def prepare_image_for_prediction(image_tensor):
    """
    Prepare a preprocessed image tensor for model prediction by adding batch dimension.
    
    Args:
        image_tensor: Preprocessed image tensor of shape (224, 224, 3)
        
    Returns:
        tf.Tensor: Image tensor with batch dimension of shape (1, 224, 224, 3)
    """
    # Add batch dimension
    return tf.expand_dims(image_tensor, axis=0)


def validate_image_format(image_bytes):
    """
    Validate that the uploaded file is a valid image format.
    
    Args:
        image_bytes: Raw image bytes from uploaded file
        
    Returns:
        bool: True if valid image format, False otherwise
    """
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Check if it's a valid image format
        image.verify()
        return True
    except Exception:
        return False