from .base_extractor import BaseExtractor
import re
import numpy as np


class GenderExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.patterns = {
            'Gender': {
                'regex': r'^[MF]$|^(male|female)$',
                'validator': lambda x: str(x).lower() in ['m', 'f', 'male', 'female'],
                'weight': 1.2,
                'invalid_patterns': [
                    r'.*Guest.*',
                    r'.*roommate.*',
                    r'.*\*.*',
                    r'^\d+$'
                ]
            }
        }

    def generate_features(self, row, header_coords):
        try:
            x_distance = abs(float(row['left']) - float(header_coords['left']))
            y_distance = float(row['top']) - float(header_coords['top'])
            euclidean_distance = np.sqrt(x_distance**2 + y_distance**2)
            
            horizontal_score = np.exp(-x_distance * 100) if x_distance <= 0.05 else 0
            vertical_score = 1.0 if 0 < y_distance < 0.7 else 0
            alignment_score = 1 if x_distance <= 0.02 else 0
            
            value_text = str(row['text'])
            pattern_score = self.validate_pattern(
                value_text, 
                'Gender'
            )
            character_count = len(value_text)
            is_integer = 1 if value_text.isdigit() else 0
            
            is_valid = 0
            if (pattern_score > 0.7 and
                horizontal_score > 0.5 and  # Menos restrictivo
                vertical_score > 0.5 and    # Menos restrictivo
                x_distance < 0.05 and      # Más permisivo
                y_distance > 0.02):        # Más permisivo
                is_valid = 1

            return {
                'value_text': value_text,
                'x_distance': x_distance,
                'y_distance': y_distance,
                'euclidean_distance': euclidean_distance,
                'horizontal_score': horizontal_score,
                'vertical_score': vertical_score,
                'alignment_score': alignment_score,
                'pattern_score': pattern_score,
                'character_count': character_count,
                'is_integer': is_integer,
                'is_valid': is_valid
            }
            
        except Exception as e:
            print(f"Error generando características: {str(e)}")
            return None