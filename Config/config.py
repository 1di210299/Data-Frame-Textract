# Config/config.py
CONFIG = {
    'thresholds': {
        'pattern_score': 0.7,
        'distance': 0.1,
        'digit_ratio': 0.5,
        'similarity_threshold': 0.8
    },
    'model_params': {
        'lightgbm': {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9
        },
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 3,
            'min_samples_split': 50,
            'min_samples_leaf': 20,
            'max_features': 2,
            'class_weight': {0: 1, 1: 3},
            'bootstrap': True,
            'max_samples': 0.5,
            'random_state': 42
        }
    },
    'paths': {
        'embedding_model': 'all-MiniLM-L6-v2',
        'model_dir': 'Model',
        'features_dir': 'Features',
        'predictions_dir': 'Predictions'
    }
}