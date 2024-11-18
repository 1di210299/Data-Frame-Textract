# Utils/feature_processor.py
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from Config.config import CONFIG

class FeatureProcessor:
    def __init__(self):
        self.embedding_model = SentenceTransformer(CONFIG['paths']['embedding_model'])
        
    def get_embeddings(self, text):
        """Genera embeddings para un texto"""
        return self.embedding_model.encode([text])[0]
        
    def normalize_coordinates(self, coords, doc_width=1.0, doc_height=1.0):
        """Normaliza coordenadas"""
        return {
            'x': coords['left'] / doc_width,
            'y': coords['top'] / doc_height
        }
        
    def calculate_spatial_features(self, value_coords, header_coords):
        """Calcula características espaciales avanzadas"""
        x_dist = abs(value_coords['x'] - header_coords['x'])
        y_dist = abs(value_coords['y'] - header_coords['y'])
        
        return {
            'x_distance': x_dist,
            'y_distance': y_dist,
            'diagonal_distance': np.sqrt(x_dist**2 + y_dist**2),
            'alignment_score': 1.0 if x_dist <= 0.02 else np.exp(-x_dist * 10),
            'vertical_relation': np.exp(-abs(y_dist - 0.1) * 5)
        }
        
    def perform_clustering(self, coordinates):
        """Realiza clustering espacial"""
        clusterer = DBSCAN(
            eps=CONFIG['thresholds']['distance'],
            min_samples=2
        )
        return clusterer.fit_predict(coordinates)