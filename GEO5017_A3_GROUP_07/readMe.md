Info for running training and validation process of waste detection.

## Dependencies:
- pandas: for best model results display
- ultralytics: for getting the model

## How to run
General notes:
- All code is expected to be run from the GEO5017_A3_GROUP_07 directory, each program can be called using their name via the commandline.
- Before running the paths in the data.yaml file need to be set correctly based on the user's paths.
Running steps:
- First **training.py** is run to train the model on the training data. Hyperparameters for the training can be set, such as the batch and image size and the number of epochs.
- After that **predictions.py** can be run to identify the top 100 most confident predictions, i.e. the P@100. The output directory should be set to match the model for which the predictions are made for clarity. This can be set as a parameter.
- Lastly, **best_model.py** determines which model, i.e. which hyperparameter combination, performs best by calculating 4 metrics and outputting them to the **final_test_comparison.csv** file.