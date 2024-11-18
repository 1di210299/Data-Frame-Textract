PATHS = {
    'json_dir': '/home/juandi/Documents/TextractModel/Data',
    'output_dir': '/home/juandi/Documents/TextractModel/Docs/Features',
    'models_dir': '/home/juandi/Documents/TextractModel/Model',
    'test_json': '/home/juandi/Documents/Textract/textract_results/doc_AM_01_06_24 Emergency Contacts and Insurance Report_page_1.png.json',
        'predictions_dir': '/home/juandi/Documents/TextractModel/Docs/Predictions'
}


MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'random_state': 42
    }
}

TRAINING_CONFIGS = {
    'test_size': 0.2,
    'random_state': 42
}