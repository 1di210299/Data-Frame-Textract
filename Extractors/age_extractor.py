from .base_extractor import BaseExtractor
import re
import numpy as np

class AgeExtractor(BaseExtractor):
   def __init__(self):
       super().__init__()
       self.patterns = {
           'Age': {
               'regex': r'^\d{1,2}$',
               'validator': lambda x: isinstance(x, str) and x.isdigit() and 0 <= int(x) <= 120,
               'weight': 1.2,
               'invalid_patterns': [
                   r'.*Guest.*',
                   r'.*roommate.*', 
                   r'.*\*.*',
                   r'.*[A-Za-z]+.*',
                   r'^0[0-9].*$',
                   r'^(1[2-9]\d|[2-9]\d{2}|[1-9]\d{3,})$'
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
           
           text_value = str(row['text'])
           pattern_score = self.validate_pattern(text_value, 'Age')
           character_count = len(text_value)
           is_integer = 1 if text_value.isdigit() else 0
           
           is_valid = 0
           if (pattern_score > 0.7 and 
               horizontal_score > 0.5 and 
               vertical_score > 0 and 
               x_distance < 0.05 and 
               0 < y_distance < 0.7 and
               character_count <= 3 and
               is_integer == 1):
               
               try:
                   age = int(text_value)
                   if 0 < age <= 120:
                       is_valid = 1
               except ValueError:
                   is_valid = 0

           return {
               'value_text': text_value,
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