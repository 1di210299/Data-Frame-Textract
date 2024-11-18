import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re
import json

class BaseExtractor():
   def __init__(self):
       self.scaler = StandardScaler()
       self.kmeans = KMeans(n_clusters=2, random_state=42)
       self.current_id = 10001
   
   def identify_table_structure(self,df):
       if df.empty:
           print('Error: DataFrame Vacio')
           return None
       try:
           df['header_key'] = df.apply(
               lambda row: f"{row['header_left']:.6f}_{row['header_top']:6f}",
               axis =1
           )
           
           header_mapping = {}
           unique_headers = df[['header_text','header_left','header_top','header_key']]
           
           for _,header in unique_headers.iterrows():
               if header['header_key'] not in header_mapping:
                   header_mapping[header['header_key']] = self.current_id
                   self.current_id += 1
                   
           df['header_id'] = df['header_key'].map(header_mapping)
           
           features = np.array([df['value_top'], df['value_left']]).T
           
           if len(features)>0:
               features_scaled = self.scaler.fit_transform(features)
               df['row_cluster'] = self.kmeans.fit_predict(features_scaled)
           else:
               df['row_cluster'] = 0
           
           return df
           
       except Exception as e:
           print(f'Error en identify_table_structure: {str(e)}')
           return None
   
   def validate_pattern(self,text,header):
       if header not in self.patterns:
           return 1.0
       
       pattern = self.patterns[header]
       
       for invalid_pattern in pattern['invalid_patterns']:
           if re.search(invalid_pattern,str(text), re.IGNORECASE):
               return 0
           
       if not re.match(pattern['regex'],str(text)):
           return 0
       
       if not pattern['validator'](str(text)):
           return 0
       
       return 1

   def extract_data(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        header_coords = None
        values = []
        
        for item in data.get('bounding_boxes', []):
            text = item.get('text', '').strip()
            geometry = item.get('geometry', {}).get('BoundingBox', {})
            
            if not text or not geometry:
                continue
            
            if text.lower() in [k.lower() for k in self.patterns.keys()]:
                header_coords = {
                    'left': float(geometry.get('Left', 0)),
                    'top': float(geometry.get('Top', 0))
                }
            elif self.validate_pattern(text, list(self.patterns.keys())[0]) > 0:
                values.append({
                    'text': text,
                    'left': float(geometry.get('Left', 0)),
                    'top': float(geometry.get('Top', 0))
                })
        
        return header_coords, values

                