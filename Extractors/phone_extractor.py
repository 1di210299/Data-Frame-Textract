from .base_extractor import BaseExtractor
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import re
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PhoneExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        # Inicializar modelo de embeddings
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error(f"Error inicializando modelo de embeddings: {e}")
            self.embedding_model = None

        self.patterns = {
            'Phone': {
                'regex': r'^\(\d{3}\)\s\d{3}-\d{4}$',  # Formato exacto (XXX) XXX-XXXX
                'header_patterns': [
                    r'.*emergency.*',
                    r'.*phone.*',
                    r'.*contact.*',
                    r'.*tel.*',  # Añadido para "telephone"
                    r'.*cell.*',  # Añadido para "cell phone"
                    r'.*mobile.*'  # Añadido para "mobile"
                ],
                'invalid_patterns': [
                    r'.*guide.*',
                    r'.*date.*',
                    r'.*january.*',
                    r'.*february.*',
                    r'.*@.*',
                    r'.*fax.*',
                    r'.*extension.*'
                ]
            }
        }
        
        self.area_codes = set()  # Conjunto para almacenar códigos de área válidos
        
    def validate_header(self, text):
        """Validación mejorada de headers"""
        if not text:
            return False
            
        text = str(text).strip().lower()
        
        # Lista expandida de palabras clave negativas
        negative_keywords = [
            'insurance',
            'report',
            'expedition',
            'guest',
            'name',
            'address',
            'email',
            'fax',
            'extension',
            'ext',
            'zip',
            'postal',
            'code'
        ]
        
        # Verificación de palabras clave negativas
        for keyword in negative_keywords:
            if keyword in text:
                return False
        
        # Verificar patrones de header
        for header_pattern in self.patterns['Phone']['header_patterns']:
            if re.match(header_pattern, text, re.IGNORECASE):
                # Verificaciones adicionales mejoradas
                
                # Control de dígitos
                digit_count = sum(c.isdigit() for c in text)
                if digit_count > 2:
                    return False
                
                # Control de longitud
                if len(text) > 30:
                    return False
                
                # Control de caracteres especiales
                special_chars = len(re.findall(r'[^\w\s]', text))
                if special_chars > 2:
                    return False
                
                # Verificación de palabras clave positivas
                positive_keywords = ['phone', 'contact', 'emergency', 'tel', 'mobile', 'cell']
                if any(keyword in text for keyword in positive_keywords):
                    return True
                    
        return False
    
    def get_embeddings(self, text):
        """Genera embeddings para un texto"""
        try:
            if self.embedding_model:
                return self.embedding_model.encode([str(text)])[0]
            return None
        except Exception as e:
            logger.error(f"Error generando embeddings: {e}")
            return None

    def calculate_semantic_similarity(self, text1, text2):
        """Calcula similitud semántica entre dos textos"""
        try:
            if self.embedding_model:
                emb1 = self.get_embeddings(text1)
                emb2 = self.get_embeddings(text2)
                if emb1 is not None and emb2 is not None:
                    return cosine_similarity([emb1], [emb2])[0][0]
            return 0.0
        except Exception as e:
            logger.error(f"Error calculando similitud semántica: {e}")
            return 0.0

    def normalize_coordinates(self, coords, doc_width=1.0, doc_height=1.0):
        """Normaliza coordenadas basadas en dimensiones del documento"""
        try:
            return {
                'x': coords['left'] / doc_width,
                'y': coords['top'] / doc_height
            }
        except Exception as e:
            logger.error(f"Error normalizando coordenadas: {e}")
            return {'x': 0.0, 'y': 0.0}

    def calculate_spatial_features(self, value_coords, header_coords):
        """Calcula características espaciales avanzadas"""
        try:
            x_dist = abs(value_coords['x'] - header_coords['x'])
            y_dist = abs(value_coords['y'] - header_coords['y'])
            
            return {
                'x_distance': x_dist,
                'y_distance': y_dist,
                'diagonal_distance': np.sqrt(x_dist**2 + y_dist**2),
                'alignment_score': 1.0 if x_dist <= 0.02 else np.exp(-x_dist * 10),
                'vertical_relation': np.exp(-abs(y_dist - 0.1) * 5)
            }
        except Exception as e:
            logger.error(f"Error calculando características espaciales: {e}")
            return {
                'x_distance': 0.0,
                'y_distance': 0.0,
                'diagonal_distance': 0.0,
                'alignment_score': 0.0,
                'vertical_relation': 0.0
            }

    def is_valid_area_code(self, area_code):
        """Validación mejorada de código de área"""
        if not area_code or len(area_code) != 3:
            return False
            
        # Verificaciones básicas
        if area_code in self.invalid_area_codes:
            return False
            
        if len(set(area_code)) == 1:  # Todos los dígitos iguales
            return False
            
        if area_code.startswith(('0', '1')):
            return False
            
        # Verificaciones adicionales
        try:
            area_code_int = int(area_code)
            if area_code_int < 200:  # Códigos muy bajos
                return False
            if area_code_int > 999:  # Códigos inválidos
                return False
        except ValueError:
            return False
            
        return True

    def validate_pattern(self, text, pattern_name, row_coords=None, header_coords=None):
        """Sistema de puntuación mejorado para validación de patrones"""
        if not text:
            return 0
                
        text = str(text).strip()
        cleaned_text = re.sub(r'[^\d]', '', text)
        score = 0
        
        # Formato básico (0.3 puntos)
        if re.match(r'^\(\d{3}\)\s\d{3}-\d{4}$', text):
            score += 0.3
        elif re.match(r'^\d{3}-\d{3}-\d{4}$', text):  # Formato alternativo
            score += 0.25
        elif re.match(r'^\d{10}$', cleaned_text):  # Solo dígitos
            score += 0.2
        
        # Longitud de dígitos (0.2 puntos)
        if len(cleaned_text) == 10:
            score += 0.2
        
        # Código de área (0.2 puntos)
        area_code = cleaned_text[:3]
        if self.is_valid_area_code(area_code):
            score += 0.2
        
        # Verificación de posición (0.3 puntos)
        if row_coords and header_coords:
            try:
                # Normalizar coordenadas
                row_norm = self.normalize_coordinates(row_coords)
                header_norm = self.normalize_coordinates(header_coords)
                
                # Calcular características espaciales
                spatial_features = self.calculate_spatial_features(row_norm, header_norm)
                
                # Asignar puntos por posición
                if spatial_features['x_distance'] <= 0.1:
                    score += 0.15
                if 0 < spatial_features['y_distance'] < 0.4:
                    score += 0.15
                    
            except Exception as e:
                logger.error(f"Error en validación de posición: {e}")
        
        return min(1.0, score)  # Normalizar a máximo 1.0

    def generate_features(self, value, header_coords):
        """Generación de características mejorada"""
        try:
            # Texto básico
            text = str(value['text']).strip()
            header_text = str(header_coords['text']).strip() if 'text' in header_coords else ''
            
            # Normalizar coordenadas y calcular características espaciales
            value_norm = self.normalize_coordinates(value)
            header_norm = self.normalize_coordinates(header_coords)
            spatial_features = self.calculate_spatial_features(value_norm, header_norm)
            
            # Características de texto
            cleaned_text = re.sub(r'[^\d]', '', text)
            digit_ratio = len(cleaned_text) / len(text) if len(text) > 0 else 0
            special_chars = len(re.findall(r'[^\w\s]', text))
            
            # Score del patrón
            pattern_score = self.validate_pattern(
                text,
                'Phone',
                row_coords=value,
                header_coords=header_coords
            )
            
            # Similitud semántica
            semantic_similarity = self.calculate_semantic_similarity(text, header_text)
            
            # Características base
            features = {
                'text': text,
                'x_distance': spatial_features['x_distance'],
                'y_distance': spatial_features['y_distance'],
                'diagonal_distance': spatial_features['diagonal_distance'],
                'alignment_score': spatial_features['alignment_score'],
                'vertical_relation': spatial_features['vertical_relation'],
                'digit_ratio': digit_ratio,
                'special_chars': special_chars,
                'pattern_score': pattern_score,
                'char_count': len(text),
                'has_parentheses': 1 if '(' in text and ')' in text else 0,
                'has_hyphen': 1 if '-' in text else 0,
                'has_space': 1 if ' ' in text else 0,
                'semantic_similarity': semantic_similarity,
                'position_score': max(0, 1 - (spatial_features['diagonal_distance'])),
                'is_valid': 1 if pattern_score >= 0.7 else 0
            }
            
            # Características adicionales
            features.update({
                'digit_sequence_score': self.evaluate_digit_sequence(cleaned_text),
                'format_consistency': self.evaluate_format_consistency(text),
                'relative_position': self.calculate_relative_position(value_norm, header_norm)
            })
            
            return features
                
        except Exception as e:
            logger.error(f"Error generando características: {e}")
            return None

    def evaluate_digit_sequence(self, cleaned_text):
        """Evalúa la secuencia de dígitos"""
        try:
            if len(cleaned_text) != 10:
                return 0.0
                
            # Verificar repeticiones
            if any(cleaned_text.count(d) > 4 for d in set(cleaned_text)):
                return 0.3
                
            # Verificar secuencias
            for i in range(len(cleaned_text)-2):
                if cleaned_text[i:i+3] in '0123456789' or cleaned_text[i:i+3] in '9876543210':
                    return 0.5
                    
            return 1.0
        except Exception:
            return 0.0

    def evaluate_format_consistency(self, text):
        """Evalúa la consistencia del formato"""
        try:
            # Formatos comunes
            formats = [
                r'^\(\d{3}\)\s\d{3}-\d{4}$',  # (XXX) XXX-XXXX
                r'^\d{3}-\d{3}-\d{4}$',        # XXX-XXX-XXXX
                r'^\d{10}$'                     # XXXXXXXXXX
            ]
            
            for fmt in formats:
                if re.match(fmt, text):
                    return 1.0
                    
            return 0.5
        except Exception:
            return 0.0

    def calculate_relative_position(self, value_norm, header_norm):
        """Calcula la posición relativa normalizada"""
        try:
            dx = value_norm['x'] - header_norm['x']
            dy = value_norm['y'] - header_norm['y']
            
            # Preferimos valores ligeramente a la derecha y abajo del header
            x_score = np.exp(-abs(dx - 0.1) * 5)
            y_score = np.exp(-abs(dy - 0.05) * 5)
            
            return (x_score + y_score) / 2
        except Exception:
            return 0.0