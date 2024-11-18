import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
from Extractors.age_extractor import AgeExtractor
from Extractors.gender_extractor import GenderExtractor 
from Extractors.name_extractor import NameExtractor
from Extractors.phone_extractor import PhoneExtractor
from Extractors.emergency_phone_extractor import EmergencyPhoneExtractor
from Config.settings import PATHS, MODEL_CONFIGS, TRAINING_CONFIGS

def load_json_data(json_path):
   with open(json_path, 'r', encoding='utf-8') as f:
       return json.load(f)

def process_files(json_dir, extractor):
  headers = []
  values = []
  debug_file = os.path.join('/home/juandi/Documents/TextractModel', 'debug_log.txt')
  
  with open(debug_file, 'a') as f:
      f.write(f"\n{'='*50}\n")
      f.write(f"\nRevisando directorio: {json_dir}\n")
      f.write(f"Patrones del extractor: {extractor.patterns.keys()}\n")
      
      for filename in os.listdir(json_dir):
          if filename.endswith('.json'):
              f.write(f"\nProcesando archivo: {filename}\n")
              file_path = os.path.join(json_dir, filename)
              data = load_json_data(file_path)
              
              f.write(f"Número de bounding boxes: {len(data.get('bounding_boxes', []))}\n")
              
              for item in data.get('bounding_boxes', []):
                  text = item.get('text', '').strip()
                  geometry = item.get('geometry', {}).get('BoundingBox', {})
                  
                  if not text or not geometry:
                      f.write(f"Texto o geometría inválida: {text}\n")
                      continue
                      
                  item_data = {
                      'text': text,
                      'left': float(geometry.get('Left', 0)),
                      'top': float(geometry.get('Top', 0)),
                      'file': filename
                  }
                  
                  f.write(f"Procesando texto: '{text}' en posición left={item_data['left']}, top={item_data['top']}\n")
                  
                  for pattern_name in extractor.patterns.keys():
                      f.write(f"Verificando patrón: {pattern_name}\n")
                      
                      if text.lower() == pattern_name.lower():
                          f.write(f"Header encontrado: {text}\n")
                          headers.append(item_data)
                          
                      validation_score = extractor.validate_pattern(text, pattern_name)
                      f.write(f"Score de validación para '{text}': {validation_score}\n")
                      
                      if validation_score > 0:
                          print(f"Encontrado teléfono válido: {text}")
                          print(f"Score: {extractor.validate_pattern(text, pattern_name)}")
                          f.write(f"Valor válido encontrado: {text}\n")
                          values.append(item_data)
      
      f.write(f"\nResumen final:\n")
      f.write(f"Headers encontrados: {len(headers)}\n")
      f.write(f"Valores encontrados: {len(values)}\n")
      f.write(f"Headers: {headers}\n")
      f.write(f"Values: {values}\n")
      f.write(f"{'='*50}\n\n")
  
  return headers, values


def generate_features(value, header_coords, extractor):
    try:
        x_distance = abs(float(value['left']) - float(header_coords['left']))
        y_distance = float(value['top']) - float(header_coords['top'])
        euclidean_distance = np.sqrt(x_distance**2 + y_distance**2)
        
        horizontal_score = np.exp(-x_distance * 100) if x_distance <= 0.05 else 0
        vertical_score = 1.0 if 0 < y_distance < 0.7 else 0
        alignment_score = 1 if x_distance <= 0.02 else 0
        
        pattern_score = extractor.validate_pattern(
            value['text'], 
            extractor.column_config['column_name']
        )
        character_count = len(str(value['text']))
        is_integer = 1 if str(value['text']).isdigit() else 0
        
        return {
            'x_distance': x_distance,
            'y_distance': y_distance,
            'euclidean_distance': euclidean_distance,
            'horizontal_score': horizontal_score,
            'vertical_score': vertical_score, 
            'alignment_score': alignment_score,
            'pattern_score': pattern_score,
            'character_count': character_count,
            'is_integer': is_integer
        }
    except Exception as e:
        print(f"Error generando características: {e}")
        return None

def process_test_file(json_path, model, extractor):
   try:
        print("\nBuscando números telefónicos...")
        header_coords, values = extractor.extract_data(json_path)
        
        if not header_coords:
            print(f"No se encontró header de teléfono. Buscando: {list(extractor.patterns.keys())}")
            return None
            
        print(f"\nEncontrados {len(values)} números potenciales")
        for v in values:
            print(f"- {v['text']}")
           
        test_features = []
        raw_data = []
       
        print(f"\nProcesando {len(values)} valores potenciales")
        
        for value in values:
            print(f"\nAnalizando valor: {value['text']}")
            feature_row = extractor.generate_features(value, header_coords)
            
            if feature_row:
                print(f"Características generadas:")
                print(f"Pattern score: {feature_row['pattern_score']}")
                print(f"Horizontal score: {feature_row['horizontal_score']}")
                print(f"Vertical score: {feature_row['vertical_score']}")
                print(f"Is valid: {feature_row['is_valid']}")
                
                test_features.append(feature_row)
                raw_data.append({
                    'text': value['text'],
                    'left': value['left'], 
                    'top': value['top']
                })
        
        if not test_features:
            print("\nNo se pudieron generar características para predicción")
            return None
            
        print(f"\nCaracterísticas generadas: {len(test_features)}")
        
        test_feature_df = pd.DataFrame(test_features)
        X_pred = test_feature_df[[
            'x_distance', 'y_distance', 'horizontal_score', 'vertical_score',
            'alignment_score', 'pattern_score', 'character_count', 'is_integer'
        ]]
        
        print("\nRealizando predicciones...")
        predictions = model.predict(X_pred)
        probabilities = model.predict_proba(X_pred)[:, 1]
        
        results_df = pd.DataFrame(raw_data)
        results_df['predicted_valid'] = predictions
        results_df['confidence'] = probabilities
        
        print("\nResultados:")
        print(results_df[['text', 'predicted_valid', 'confidence']])
        
        return results_df
        
   except Exception as e:
        print(f'Error en process_test_file: {e}')
        return None

def train_model(feature_df):
   required_columns = [
       'x_distance', 'y_distance', 'horizontal_score', 'vertical_score',
       'alignment_score', 'pattern_score', 'character_count', 'is_integer'
   ]
   
   X = feature_df[required_columns]
   y = feature_df['is_valid']
   
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, **TRAINING_CONFIGS
   )
   
   model = RandomForestClassifier(**MODEL_CONFIGS['random_forest'])
   model.fit(X_train, y_train)
   
   y_pred = model.predict(X_test)
   print("\nResultados de evaluación:")
   print("Accuracy:", accuracy_score(y_test, y_pred))
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))
   
   return model

def main():
   extractors = {

       'phone': PhoneExtractor(),
   }

   # Crear directorios si no existen
   os.makedirs(PATHS['output_dir'], exist_ok=True)
   os.makedirs(PATHS['predictions_dir'], exist_ok=True)
   os.makedirs(PATHS['models_dir'], exist_ok=True)

   for name, extractor in extractors.items():
       print(f"\n{'='*50}")
       print(f"Procesando {name.upper()}")
       print(f"Patrones configurados: {extractor.patterns}")
       
       # Procesar archivos
       headers, values = process_files(PATHS['json_dir'], extractor)
       
       if not headers or not values:
           print(f"No se encontraron datos para {name}")
           print(f"Headers: {headers}")
           print(f"Values: {values}")
           continue
           
       # Generar características
       feature_data = []
       for header in headers:
           for value in values:
               if value['file'] == header['file']:
                   features = extractor.generate_features(value, header)
                   if features:
                       feature_data.append(features)
       
       if not feature_data:
           print(f"No se pudieron generar características para {name}")
           continue
           
       # Crear DataFrame y entrenar modelo
       feature_df = pd.DataFrame(feature_data)
       model = train_model(feature_df)
       
       # Guardar modelo y características
       model_path = os.path.join(PATHS['models_dir'], f'{name}_model.pkl')
       output_path = os.path.join(PATHS['output_dir'], f'{name}_features.csv')
       
       joblib.dump(model, model_path)
       feature_df.to_csv(output_path, index=False)
       print(f"Features guardadas en: {output_path}")
       print(f"Modelo guardado en: {model_path}")
       
       # Procesar archivo de prueba
       test_results = process_test_file(PATHS['test_json'], model, extractor)
       
       if test_results is not None:
           print("\nPredicciones finales:")
           valid_predictions = test_results[test_results['predicted_valid'] == 1]
           print("\nValores identificados como válidos:")
           print(valid_predictions[['text', 'confidence']].sort_values('confidence', ascending=False))
           
           test_results_path = os.path.join(PATHS['predictions_dir'], f'{name}_predictions.csv')
           test_results.to_csv(test_results_path, index=False)
           print(f"\nPredicciones guardadas en: {test_results_path}")
       
       print(f"\nProcesamiento de {name} completado")

if __name__ == "__main__":
   main()