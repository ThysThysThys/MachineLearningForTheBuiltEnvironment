from ultralytics import YOLO
import os
import pandas as pd

# 1. Setup paths
data_yaml = 'GEO5017-A3-UrbanWaste/data.yaml'  # Ensure 'test:' path is in here
runs_dir = 'runs/detect'
# Get your 5 model folders
model_folders = [d for d in os.listdir(runs_dir) if 'UrbanWaste' in d]

test_results = []

for folder in model_folders:
    model_path = os.path.join(runs_dir, folder, 'weights', 'best.pt')
    if not os.path.exists(model_path): continue
    
    print(f"🚀 Evaluating: {folder}")
    model = YOLO(model_path)
    
    # Run validation on the TEST split using your annotations
    metrics = model.val(data=data_yaml, split='test', device='mps', plots=True, name=f"Final_Test_{folder}")
    
    # Collect data for your report
    test_results.append({
        "Model Name": folder,
        "Precision": metrics.results_dict['metrics/precision(B)'],
        "Recall": metrics.results_dict['metrics/recall(B)'],
        "mAP50": metrics.results_dict['metrics/mAP50(B)'],
        "Fitness": metrics.fitness
    })

# 2. Display the Leaderboard
df = pd.DataFrame(test_results).sort_values(by='Precision', ascending=False)
print("\n🏆 TEST SET LEADERBOARD (Ranked by Precision) 🏆")
print(df)
df.to_csv('final_test_comparison.csv', index=False)