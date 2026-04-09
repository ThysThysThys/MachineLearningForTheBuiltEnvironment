from ultralytics import YOLO
import os
import pandas as pd

# parameters
# path to data yaml, ensure 'test:' path is in here
data_yaml = 'data.yaml'
runs_dir = '../runs/detect'
# for GPU running
runs_on_mac = True

if __name__ == "__main__":
    # get folders of models with different hyperparameters to be checked
    model_folders = [d for d in os.listdir(runs_dir) if 'UrbanWaste' in d]

    test_results = []

    for folder in model_folders:
        # get parameters and weights for model
        model_path = os.path.join(runs_dir, folder, 'weights', 'best.pt')
        if not os.path.exists(model_path): 
            continue
        
        # determine correct GPU running settinga
        if runs_on_mac:
            device = 'mps'
        else:
            device = 0

        # load model with retrieved parameters and weights
        model = YOLO(model_path)
        
        # run validation on the test data with annotations for confusion matrix and further analysis
        metrics = model.val(data=data_yaml, split='test', device=device, plots=True, name=f"Final_Test_{folder}")
        
        # store results
        test_results.append({
            "Model Name": folder,
            "Precision": metrics.results_dict['metrics/precision(B)'],
            "Recall": metrics.results_dict['metrics/recall(B)'],
            "mAP50": metrics.results_dict['metrics/mAP50(B)'],
            "Fitness": metrics.fitness
        })

    # print and save results of models
    df = pd.DataFrame(test_results).sort_values(by='Precision', ascending=False)
    print("Best models ranked by precision")
    print(df)
    df.to_csv('final_test_comparison.csv', index=False)