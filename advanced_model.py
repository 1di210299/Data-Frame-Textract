# Model/advanced_model.py
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import numpy as np
from Config.config import CONFIG

class AdvancedPhoneModel:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def train(self, X, y):
        """Entrena el modelo usando LightGBM"""
        train_data = lgb.Dataset(X, label=y)
        
        self.model = lgb.train(
            CONFIG['model_params']['lightgbm'],
            train_data,
            num_boost_round=100
        )
        self.feature_names = X.columns.tolist()
        
    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)
        
    def evaluate(self, X, y):
        """Evalúa el modelo con validación cruzada"""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Entrenar en este fold
            train_data = lgb.Dataset(X_train, label=y_train)
            model = lgb.train(CONFIG['model_params']['lightgbm'], train_data)
            
            # Predecir
            y_pred = model.predict(X_val)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calcular métricas
            fold_scores = {
                'auc': roc_auc_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred_binary),
                'recall': recall_score(y_val, y_pred_binary),
                'f1': f1_score(y_val, y_pred_binary)
            }
            scores.append(fold_scores)
            
        return {k: np.mean([s[k] for s in scores]) for k in scores[0].keys()}