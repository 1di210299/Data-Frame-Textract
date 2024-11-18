import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np
from Extractors.phone_extractor import PhoneExtractor
from datetime import datetime
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import StratifiedKFold  # Esta falta
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

FEATURE_COLUMNS = [
    'x_distance',
    'y_distance',
    'digit_ratio',
    'special_chars',
    'pattern_score',
    'char_count',
    'has_parentheses',
    'has_hyphen',
    'has_space',
    'position_score'
]

PATHS = {
    'json_dir': os.path.join(os.path.dirname(__file__), 'Data'),
    'output_dir': os.path.join(os.path.dirname(__file__), 'Features'),
    'models_dir': os.path.join(os.path.dirname(__file__), 'Model'),
    'predictions_dir': os.path.join(os.path.dirname(__file__), 'Predictions'),
    'test_json': r'/home/juandi/Documents/Textract/textract_results/doc_AM_01_06_24 Emergency Contacts and Insurance Report_page_1.png.json'  # Agregamos esta línea
}

# Crear directorios si no existen
for key, path in PATHS.items():
    if key != 'test_json':  # Excluir archivos específicos como el archivo de prueba
        os.makedirs(path, exist_ok=True)

def process_test_file(file_path, model, extractor):
    test_results_file = os.path.join(os.path.dirname(__file__), 'test_results.txt')
    predictions_file = os.path.join(PATHS['predictions_dir'], 'phone_predictions.csv')
    
    with open(test_results_file, 'w', encoding='utf-8') as f:
        f.write("=== RESULTADOS DE PRUEBA DE EXTRACCIÓN DE TELÉFONOS ===\n\n")
        f.write(f"Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Archivo procesado: {file_path}\n\n")
        
        data = load_json_data(file_path)
        if not data:
            f.write("ERROR: No se pudo cargar el archivo\n")
            return None
            
        test_features = []
        headers = []
        values = []
        
        # Procesamiento de bounding boxes
        total_boxes = len(data.get('bounding_boxes', []))
        f.write(f"Total de bounding boxes analizados: {total_boxes}\n\n")
        
        # Primero encontrar headers
        f.write("=== HEADERS ENCONTRADOS ===\n")
        for item in data.get('bounding_boxes', []):
            text = item.get('text', '').strip()
            geometry = item.get('geometry', {}).get('BoundingBox', {})
            
            if not text or not geometry:
                continue
                
            item_data = {
                'text': text,
                'left': float(geometry.get('Left', 0)),
                'top': float(geometry.get('Top', 0))
            }
            
            if extractor.validate_header(text):
                f.write(f"Header: '{text}'\n")
                f.write(f"Posición: left={item_data['left']:.3f}, top={item_data['top']:.3f}\n")
                headers.append(item_data)
        
        f.write(f"\nTotal headers encontrados: {len(headers)}\n\n")
        
        # Luego buscar números
        f.write("=== NÚMEROS ENCONTRADOS ===\n")
        for item in data.get('bounding_boxes', []):
            text = item.get('text', '').strip()
            geometry = item.get('geometry', {}).get('BoundingBox', {})
            
            if not text or not geometry:
                continue
                
            item_data = {
                'text': text,
                'left': float(geometry.get('Left', 0)),
                'top': float(geometry.get('Top', 0))
            }
            
            cleaned_text = re.sub(r'[^\d]', '', text)
            if len(cleaned_text) >= 7:  # Si tiene al menos 7 dígitos
                f.write(f"\nTexto encontrado: '{text}'\n")
                f.write(f"Posición: left={item_data['left']:.3f}, top={item_data['top']:.3f}\n")
                values.append(item_data)
        
        f.write(f"\nTotal números potenciales encontrados: {len(values)}\n\n")
        
        # Generar predicciones
        f.write("=== PREDICCIONES ===\n")
        predictions_data = []
        
        for header in headers:
            for value in values:
                features = generate_features(value, header, extractor)
                if features:
                    # Preparar características para predicción
                    X_test = pd.DataFrame([features])[model.feature_names_in_]
                    
                    # Realizar predicción
                    pred = model.predict(X_test)[0]
                    prob = model.predict_proba(X_test)[0][1]
                    
                    # Guardar resultados
                    result = {
                        'text': value['text'],
                        'header': header['text'],
                        'x_distance': features['x_distance'],
                        'y_distance': features['y_distance'],
                        'predicted': pred,
                        'confidence': prob
                    }
                    predictions_data.append(result)
                    
                    # Escribir en archivo de resultados
                    f.write(f"\nTexto: {value['text']}\n")
                    f.write(f"Header relacionado: {header['text']}\n")
                    f.write(f"Distancia X: {features['x_distance']:.3f}\n")
                    f.write(f"Distancia Y: {features['y_distance']:.3f}\n")
                    f.write(f"Predicción: {'Válido' if pred == 1 else 'Inválido'}\n")
                    f.write(f"Confianza: {prob:.3f}\n")
        
        # Resumen final
        if predictions_data:
            df_predictions = pd.DataFrame(predictions_data)
            valid_predictions = df_predictions[df_predictions['predicted'] == 1]
            
            f.write("\n=== RESUMEN FINAL ===\n")
            f.write(f"Total de predicciones realizadas: {len(predictions_data)}\n")
            f.write(f"Números identificados como válidos: {len(valid_predictions)}\n\n")
            
            f.write("Teléfonos válidos encontrados:\n")
            for _, row in valid_predictions.iterrows():
                f.write(f"- {row['text']} (confianza: {row['confidence']:.3f})\n")
            
            # Guardar predicciones en CSV
            df_predictions.to_csv(predictions_file, index=False)
            f.write(f"\nPredicciones detalladas guardadas en: {predictions_file}\n")
        else:
            f.write("\nNo se pudieron generar predicciones\n")
        
        f.write("\n=== FIN DEL REPORTE ===\n")
    
    print(f"\nResultados guardados en: {test_results_file}")
    if predictions_data:
        print(f"Predicciones guardadas en: {predictions_file}")
        return pd.DataFrame(predictions_data)
    return None


def load_json_data(json_path):
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error cargando archivo JSON {json_path}: {e}")
        return None


def process_files(json_dir, extractor):
    headers = []
    values = []
    
    print("\nProcesando archivos para extraer teléfonos...")
    
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            print(f"\nProcesando archivo: {filename}")
            file_path = os.path.join(json_dir, filename)
            data = load_json_data(file_path)
            
            if not data:
                print(f"Error cargando {filename}")
                continue
            
            boxes = data.get('bounding_boxes', [])
            print(f"Bounding boxes encontrados: {len(boxes)}")
            
            for item in boxes:
                text = item.get('text', '').strip()
                geometry = item.get('geometry', {}).get('BoundingBox', {})
                
                if not text or not geometry:
                    continue
                
                # Imprimir cada texto para debug
                print(f"Analizando texto: '{text}'")
                
                item_data = {
                    'text': text,
                    'left': float(geometry.get('Left', 0)),
                    'top': float(geometry.get('Top', 0)),
                    'file': filename
                }
                
                # Buscar headers
                if any(re.match(pattern, text, re.IGNORECASE) 
                      for pattern in extractor.patterns['Phone']['header_patterns']):
                    print(f"Header encontrado: {text}")
                    headers.append(item_data)
                    continue
                
                # Criterios más permisivos para identificar potenciales teléfonos
                cleaned_text = re.sub(r'[^\d]', '', text)
                if len(cleaned_text) >= 7:  # Si tiene al menos 7 dígitos
                    digit_ratio = len(cleaned_text) / len(text) if len(text) > 0 else 0
                    if digit_ratio > 0.5:  # Si más del 50% son dígitos
                        print(f"Valor potencial encontrado: {text}")
                        values.append(item_data)
    
    print(f"\nTotal headers encontrados: {len(headers)}")
    print(f"Total valores potenciales encontrados: {len(values)}")
    
    return headers, values

def generate_features(value, header_coords, extractor):
    try:
        text = str(value['text']).strip()
        x_distance = abs(float(value['left']) - float(header_coords['left']))
        y_distance = float(value['top']) - float(header_coords['top'])
        
        # Características del texto
        cleaned_text = re.sub(r'[^\d]', '', text)
        digit_ratio = len(cleaned_text) / len(text) if len(text) > 0 else 0
        special_chars = len(re.findall(r'[^\w\s]', text))
        
        # Score del patrón
        pattern_score = extractor.validate_pattern(
            text,
            'Phone',
            row_coords=value,
            header_coords=header_coords
        )
        
        features = {
            'text': text,
            'x_distance': x_distance,
            'y_distance': y_distance,
            'digit_ratio': digit_ratio,
            'special_chars': special_chars,
            'pattern_score': pattern_score,
            'character_count': len(text),           # Antes era char_count
            'has_parentheses': 1 if '(' in text and ')' in text else 0,
            'has_hyphen': 1 if '-' in text else 0,
            'has_space': 1 if ' ' in text else 0,
            'position_score': max(0, 1 - (x_distance + y_distance)/2),
            'horizontal_score': np.exp(-x_distance * 100) if x_distance <= 0.05 else 0,
            'vertical_score': 1.0 if 0 < y_distance < 0.7 else 0,
            'alignment_score': 1 if x_distance <= 0.02 else 0,
            'normalized_distance': x_distance / (y_distance + 1e-6),
            'is_integer': 1 if cleaned_text.isdigit() else 0,
            'is_valid': 1 if pattern_score >= 0.7 else 0
        }
        
        return features
        
    except Exception as e:
        print(f"Error generando características: {e}")
        return None
    
def train_model(feature_df):
    results_file = os.path.join(os.path.dirname(__file__), 'model_results.txt')
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=== RESULTADOS DEL MODELO ===\n\n")
        
        # Usar solo las características más importantes
        primary_features = [
            'pattern_score',
            'digit_ratio',
            'character_count',
            'special_chars',
            'x_distance'
        ]
        
        available_columns = [col for col in primary_features if col in feature_df.columns]
        
        f.write("Características utilizadas:\n")
        f.write(", ".join(available_columns) + "\n\n")
        
        X = feature_df[available_columns]
        y = feature_df['is_valid']
        
        # Distribución de clases
        f.write("Distribución de clases:\n")
        class_dist = y.value_counts()
        f.write(f"No válidos (0): {class_dist[0]}\n")
        f.write(f"Válidos (1): {class_dist[1]}\n")
        f.write(f"Proporción de válidos: {y.mean():.2%}\n\n")
        
        # Modelo extremadamente restrictivo
        model = RandomForestClassifier(
            n_estimators=100,          # Menos árboles
            max_depth=3,               # Profundidad muy limitada
            min_samples_split=50,      # Muchas muestras para split
            min_samples_leaf=20,       # Muchas muestras en hojas
            max_features=2,            # Solo 2 características por split
            class_weight={
                0: 1,
                1: 3                   # Mayor peso a positivos
            },
            bootstrap=True,
            max_samples=0.5,           # Solo 50% de muestras por árbol
            random_state=42,
            n_jobs=-1
        )
        
        # Validación cruzada más rigurosa
        cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=3,
            random_state=42
        )
        
        # Múltiples métricas
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        
        cv_results = cross_validate(
            model,
            X, 
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=True
        )
        
        f.write("Resultados de validación cruzada:\n")
        for metric in scoring:
            train_scores = cv_results[f'train_{metric}']
            test_scores = cv_results[f'test_{metric}']
            f.write(f"\n{metric.capitalize()}:\n")
            f.write(f"Train: {train_scores.mean():.3f} (±{train_scores.std()*2:.3f})\n")
            f.write(f"Test: {test_scores.mean():.3f} (±{test_scores.std()*2:.3f})\n")
            
        # Split y entrenamiento final
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluación con umbral ajustado
        y_prob = model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.3, 0.9, 0.1)
        
        f.write("\nEvaluación con diferentes umbrales:\n")
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            f.write(f"\nUmbral: {threshold:.1f}\n")
            f.write(f"Accuracy: {acc:.3f}\n")
            f.write(f"Precision: {prec:.3f}\n")
            f.write(f"Recall: {rec:.3f}\n")
            f.write(f"F1: {f1:.3f}\n")
            
        # Usar mejor umbral encontrado
        best_threshold = 0.5  # Ajustar según resultados
        y_pred = (y_prob >= best_threshold).astype(int)
        
        f.write("\nResultados finales:\n")
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
        f.write("\nMatriz de confusión:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))
        
        # Importancia de características
        importances = pd.DataFrame({
            'feature': available_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        f.write("\nImportancia de características:\n")
        for _, row in importances.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")
            
        # Ejemplos de predicciones
        f.write("\nEjemplos de predicciones con probabilidades:\n")
        sample_indices = np.random.choice(len(y_test), min(10, len(y_test)), replace=False)
        for idx in sample_indices:
            text = feature_df.iloc[X_test.index[idx]]['text']
            f.write(f"\nTexto: {text}\n")
            f.write(f"Real: {y_test.iloc[idx]}, ")
            f.write(f"Predicho: {y_pred[idx]}, ")
            f.write(f"Probabilidad: {y_prob[idx]:.3f}\n")
        
        f.write("\n=== FIN DEL REPORTE ===\n")
        
    print(f"\nResultados guardados en: {results_file}")
    return model
    
def cross_validate_model(X, y, model):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }
    
    scores = cross_validate(model, X, y, 
                          scoring=scoring,
                          cv=cv, 
                          return_train_score=True)
    
    return scores

def analyze_errors(y_true, y_pred, texts, probas):
    false_positives = []
    false_negatives = []
    
    for i, (true, pred, text, prob) in enumerate(zip(y_true, y_pred, texts, probas)):
        if true == 0 and pred == 1:
            false_positives.append((text, prob))
        elif true == 1 and pred == 0:
            false_negatives.append((text, prob))
    
    print("\nFalsos Positivos:")
    for text, prob in false_positives[:10]:
        print(f"Texto: {text}, Probabilidad: {prob:.3f}")
        
    print("\nFalsos Negativos:")
    for text, prob in false_negatives[:10]:
        print(f"Texto: {text}, Probabilidad: {prob:.3f}")

def evaluate_edge_cases(model, extractor):
    print("\n=== EVALUACIÓN DE CASOS EXTREMOS ===")
    
    # Obtener las características que el modelo espera
    model_features = getattr(model, 'feature_names_in_', FEATURE_COLUMNS)
    
    edge_cases = [
        {'text': '703-470-7547', 'left': 0.5, 'top': 0.5},  # Caso válido
        {'text': '(703) 220-7430', 'left': 0.5, 'top': 0.5},  # Caso válido
        {'text': '7034707547', 'left': 0.5, 'top': 0.5},  # Caso válido
        {'text': '123-456-7890', 'left': 0.5, 'top': 0.5},  # Caso inválido
        {'text': '000-000-0000', 'left': 0.5, 'top': 0.5},  # Caso inválido
        {'text': 'abc-def-ghij', 'left': 0.5, 'top': 0.5},  # Caso inválido
    ]
    
    header_coords = {'left': 0.4, 'top': 0.4}
    results = []
    
    for case in edge_cases:
        try:
            features = generate_features(case, header_coords, extractor)
            if not features:
                print(f"No se pudieron generar características para: {case['text']}")
                continue
                
            features_df = pd.DataFrame([{col: features.get(col, 0) for col in model_features}])
            
            pred = model.predict(features_df)[0]
            prob = model.predict_proba(features_df)[0][1]
            
            results.append({
                'text': case['text'],
                'prediction': pred,
                'probability': prob,
                'pattern_score': features.get('pattern_score', 0)
            })
            
        except Exception as e:
            print(f"Error procesando caso {case['text']}: {e}")
            continue
    
    print("\nResultados de casos extremos:")
    for result in results:
        print(f"\nTexto: {result['text']}")
        print(f"Predicción: {'Válido' if result['prediction'] == 1 else 'Inválido'}")
        print(f"Probabilidad: {result['probability']:.3f}")
        print(f"Pattern Score: {result['pattern_score']}")

def main():
    print("\n=== INICIANDO PROGRAMA ===")
    print("Inicializando extractor...")
    extractor = PhoneExtractor()
    print(f"Patrones de extractor: {extractor.patterns.keys()}")
    
    print("\nProcesando archivos...")
    headers, values = process_files(PATHS['json_dir'], extractor)
    
    print("\n=== VERIFICACIÓN DE DATOS ===")
    print(f"Headers encontrados: {len(headers)}")
    if headers:
        print("\nEjemplos de headers:")
        for h in headers[:5]:
            print(f"- {h['text']}")
            
    print(f"\nValores encontrados: {len(values)}")
    if values:
        print("\nEjemplos de valores:")
        for v in values[:5]:
            print(f"- {v['text']}")
    
    # ... resto del código ...
    
    if not headers or not values:
        print("\n¡ERROR! No se encontraron datos suficientes")
        print("Por favor revise phone_extraction_log.txt para más detalles")
        return
    
    print("\nGenerando características...")
    feature_data = []
    for header in headers:
        for value in values:
            if value['file'] == header['file']:
                features = generate_features(value, header, extractor)
                if features:
                    features['text'] = value['text']
                    feature_data.append(features)
    
    if not feature_data:
        print("¡ERROR! No se pudieron generar características")
        return
    
    print("\nCreando DataFrame...")
    feature_df = pd.DataFrame(feature_data)
    
    # Redefinir las columnas que realmente necesitamos
    required_columns = ['text'] + [
        'x_distance',
        'y_distance',
        'digit_ratio',
        'special_chars',
        'pattern_score',
        'character_count',
        'horizontal_score',
        'vertical_score',
        'alignment_score',
        'normalized_distance',
        'is_integer',
        'is_valid'
    ]
    
    # Verificar qué columnas están disponibles
    available_columns = []
    for col in required_columns:
        if col in feature_df.columns:
            available_columns.append(col)
        else:
            print(f"Advertencia: Columna {col} no encontrada")
    
    # Usar solo las columnas disponibles
    feature_df = feature_df[available_columns]
    
    print("\nGuardando features...")
    output_path = os.path.join(PATHS['output_dir'], 'phone_features.csv')
    feature_df.to_csv(output_path, index=False)
    print(f"Features guardadas en: {output_path}")
    
    print("\nEntrenando modelo...")
    model = train_model(feature_df)
    evaluate_edge_cases(model, extractor)

    # Mostrar importancia de características después de entrenar el modelo
    importances = model.feature_importances_
    feature_names = ['x_distance', 'y_distance', 'horizontal_score', 'vertical_score',
                     'alignment_score', 'pattern_score', 'character_count', 'is_integer']
    print("\nImportancia de características:")
    for name, importance in zip(feature_names, importances):
        print(f"{name}: {importance}")
    
    model_path = os.path.join(PATHS['models_dir'], 'phone_model.pkl')
    joblib.dump(model, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Procesar archivo de prueba específico
    if os.path.exists(PATHS['test_json']):
        print("\nProcesando archivo de prueba...")
        test_results = process_test_file(PATHS['test_json'], model, extractor)
        if test_results is not None:
            print("\nProcesamiento de prueba completado")
            print("Revisa phone_test_log.txt para ver los resultados detallados")
            
            # Guardar predicciones
            predictions_path = os.path.join(PATHS['predictions_dir'], 'phone_predictions.csv')
            test_results.to_csv(predictions_path, index=False)
            print(f"Predicciones guardadas en: {predictions_path}")
    else:
        print(f"\nNo se encontró el archivo de prueba: {PATHS['test_json']}")
        # Al final del main()
    print("\n=== VALIDACIÓN DE RESULTADOS ===")
    
    # Revisar predicciones
    predictions_df = pd.read_csv(os.path.join(PATHS['predictions_dir'], 'phone_predictions.csv'))
    valid_phones = predictions_df[predictions_df['predicted'] == 1]
    
    print(f"\nTeléfonos detectados como válidos: {len(valid_phones)}")
    print("\nEjemplos de teléfonos válidos:")
    for _, row in valid_phones.head().iterrows():
        print(f"- {row['text']} (confianza: {row['confidence']:.2f})")
        
        
    print("\n=== PROCESO COMPLETADO ===")

   
if __name__ == "__main__":
    main()