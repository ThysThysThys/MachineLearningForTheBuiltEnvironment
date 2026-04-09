The code can be run via main.py in the same folder as this README.
Run it from the GEO5017-A2-Classification directory, otherwise the point cloud data cannot be found.
The pipeline includes feature extraction, normalisation, feature selection, and classification using SVM and Random Forest.

The models can either:
1. Automatically find the best hyperparameters and features (chosen_model = False), or  
2. Use predefined (chosen) settings (chosen_model = True).

The script outputs accuracy, F1-scores, confusion matrices, and a learning curve.