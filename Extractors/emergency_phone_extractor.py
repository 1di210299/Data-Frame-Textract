from .base_extractor import BaseExtractor
import re
import numpy as np

class EmergencyPhoneExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.patterns = {
            'Emergency Phone': {
                'regex': r'^(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}$',
                'validator': lambda x: isinstance(x, str) and re.match(r'^(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}$', x),
                'weight': 1.2,
                'invalid_patterns': [
                    r'.*Guest.*',
                    r'.*roommate.*', 
                    r'.*\*.*',
                    r'.*[A-Za-z]+.*'
                ]
            }
        }

    def validate_pattern(self, text, header):
        if header not in self.patterns:
            return 1.0

        cleaned_text = re.sub(r'[^\d]', '', str(text))
        
        pattern = self.patterns[header]
        for invalid_pattern in pattern['invalid_patterns']:
            if re.search(invalid_pattern, str(text), re.IGNORECASE):
                return 0
                
        if len(cleaned_text) != 10:
            return 0
            
        if not re.match(pattern['regex'], str(text)):
            return 0
            
        return 1

    def generate_features(self, row, header_coords):
        try:
            x_distance = abs(float(row['left']) - float(header_coords['left']))
            y_distance = float(row['top']) - float(header_coords['top'])
            euclidean_distance = np.sqrt(x_distance**2 + y_distance**2)
            
            horizontal_score = np.exp(-x_distance * 100) if x_distance <= 0.05 else 0
            vertical_score = 1.0 if 0 < y_distance < 0.7 else 0
            alignment_score = 1 if x_distance <= 0.02 else 0
            
            value_text = str(row['text'])
            pattern_score = self.validate_pattern(value_text, 'Emergency Phone')
            character_count = len(value_text)
            is_integer = 1 if re.sub(r'[^\d]', '', value_text).isdigit() else 0
            
            is_valid = 0
            if (pattern_score > 0 and
                horizontal_score > 0.75 and
                vertical_score == 1.0 and 
                x_distance < 0.003 and
                y_distance > 0.03 and
                y_distance < 0.7 and
                len(re.sub(r'[^\d]', '', value_text)) == 10):
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