"""
Model service module for Dog Vision AI backend.

This module handles loading the trained Keras model and making predictions.
"""

import tensorflow as tf
import numpy as np
import os
from typing import Tuple


class ModelService:
    """Service class for handling model loading and predictions."""
    
    def __init__(self, model_path: str):
        """
        Initialize the model service.
        
        Args:
            model_path: Path to the trained Keras model file
        """
        self.model_path = model_path
        self.model = None
        self.dog_breeds = self._get_dog_breeds()
        self.is_loaded = False
    
    def _get_dog_breeds(self):
        """
        Return the list of 120 dog breeds that the model was trained on.
        
        Note: This is a placeholder list. In a real implementation, 
        this should match the exact breeds and order from the training data.
        """
        return [
            "affenpinscher", "afghan_hound", "african_hunting_dog", "airedale",
            "american_staffordshire_terrier", "appenzeller", "australian_terrier",
            "basenji", "basset", "beagle", "bedlington_terrier", "bernese_mountain_dog",
            "black-and-tan_coonhound", "blenheim_spaniel", "bloodhound", "bluetick",
            "border_collie", "border_terrier", "borzoi", "boston_bull",
            "bouvier_des_flandres", "boxer", "brabancon_griffon", "briard",
            "brittany_spaniel", "bull_mastiff", "cairn", "cardigan",
            "chesapeake_bay_retriever", "chihuahua", "chow", "clumber",
            "cocker_spaniel", "collie", "curly-coated_retriever", "dandie_dinmont",
            "dhole", "dingo", "doberman", "english_foxhound",
            "english_setter", "english_springer", "entlebucher", "eskimo_dog",
            "flat-coated_retriever", "french_bulldog", "german_shepherd",
            "german_short-haired_pointer", "giant_schnauzer", "golden_retriever",
            "gordon_setter", "great_dane", "great_pyrenees", "greater_swiss_mountain_dog",
            "groenendael", "ibizan_hound", "irish_setter", "irish_terrier",
            "irish_water_spaniel", "irish_wolfhound", "italian_greyhound", "japanese_spaniel",
            "keeshond", "kelpie", "kerry_blue_terrier", "komondor",
            "kuvasz", "labrador_retriever", "lakeland_terrier", "leonberg",
            "lhasa", "malamute", "malinois", "maltese_dog",
            "mexican_hairless", "miniature_pinscher", "miniature_poodle", "miniature_schnauzer",
            "newfoundland", "norfolk_terrier", "norwegian_elkhound", "norwich_terrier",
            "old_english_sheepdog", "otterhound", "papillon", "pekinese",
            "pembroke", "pomeranian", "pug", "redbone",
            "rhodesian_ridgeback", "rottweiler", "saint_bernard", "saluki",
            "samoyed", "schipperke", "scotch_terrier", "scottish_deerhound",
            "sealyham_terrier", "shetland_sheepdog", "shih-tzu", "siberian_husky",
            "silky_terrier", "soft-coated_wheaten_terrier", "staffordshire_bullterrier",
            "standard_poodle", "standard_schnauzer", "sussex_spaniel", "tibetan_mastiff",
            "tibetan_terrier", "toy_poodle", "toy_terrier", "vizsla",
            "walker_hound", "weimaraner", "welsh_springer_spaniel", "west_highland_white_terrier",
            "whippet", "wire-haired_fox_terrier", "yorkshire_terrier", "basenji",
            "bearded_collie", "belgian_malinois", "belgian_sheepdog", "belgian_tervuren",
            "bernese_mountain_dog", "bichon_frise", "black_and_tan_coonhound", "bloodhound",
            "blue_heeler", "border_collie", "borzoi", "boston_terrier",
            "bouvier_des_flandres", "boxer", "briard", "brittany",
            "bull_terrier", "bulldog", "bullmastiff", "cairn_terrier"
        ]
    
    def load_model(self) -> bool:
        """
        Load the trained Keras model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file not found: {self.model_path}")
                return False
            
            print(f"Loading model from: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            self.is_loaded = True
            print("Model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self.is_loaded = False
            return False
    
    def predict(self, image_tensor: tf.Tensor) -> Tuple[str, float]:
        """
        Make a prediction on a preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor with batch dimension (1, 224, 224, 3)
            
        Returns:
            Tuple[str, float]: Predicted breed name and confidence score
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            # Make prediction
            predictions = self.model.predict(image_tensor)
            
            # Get the predicted class index
            predicted_class_index = np.argmax(predictions[0])
            
            # Get the confidence score
            confidence = float(predictions[0][predicted_class_index])
            
            # Get the breed name
            if predicted_class_index < len(self.dog_breeds):
                predicted_breed = self.dog_breeds[predicted_class_index]
            else:
                predicted_breed = f"unknown_breed_{predicted_class_index}"
            
            return predicted_breed, confidence
            
        except Exception as e:
            raise RuntimeError(f"Error during prediction: {str(e)}")
    
    def get_health_status(self) -> dict:
        """
        Get the health status of the model service.
        
        Returns:
            dict: Health status information
        """
        return {
            "model_loaded": self.is_loaded,
            "model_path": self.model_path,
            "num_classes": len(self.dog_breeds),
            "status": "healthy" if self.is_loaded else "model_not_loaded"
        }