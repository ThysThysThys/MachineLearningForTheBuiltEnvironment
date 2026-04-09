import os
import pandas as pd
from ultralytics import YOLO

# 1. Configuration
test_data_path = 'GEO5017-A3-UrbanWaste/data.yaml' # Points to your 'val' or 'test' split
runs_dir = 'runs/detect'
experiments = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d)) and 'UrbanWaste' in d]

leaderboard = []

print(f"🧐 Analyzing {len(experiments)} experiments...")

for run in experiments:
    model_path = os.path.join(runs_dir, run, 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        continue
        
    print(f"\n--- Evaluating Model: {run} ---")
    model = YOLO(model_path)
    
    # Run validation on the test/val split
    metrics = model.val(data=test_data_path, split='val', device='mps', plots=False)
    
    # Extract key metrics for your 4-page report
    precision = metrics.results_dict['metrics/precision(B)']
    recall = metrics.results_dict['metrics/recall(B)']
    mAP50 = metrics.results_dict['metrics/mAP50(B)']
    
    leaderboard.append({
        'Run Name': run,
        'Precision': round(precision, 4),
        'Recall': round(recall, 4),
        'mAP50': round(mAP50, 4),
        'Fitness Score': round(precision * 0.7 + mAP50 * 0.3, 4) # Weighting Precision higher for your goal
    })

# 2. Rank and Save Results
df = pd.DataFrame(leaderboard)
df = df.sort_values(by='Fitness Score', ascending=False)

print("\n🏆 FINAL LEADERBOARD 🏆")
print(df.to_string(index=False))

# Save for your report
df.to_csv('model_comparison_results.csv', index=False)

# 3. Get the absolute "Best" model
best_run = df.iloc[0]['Run Name']
print(f"\n🥇 BEST MODEL: {best_run}")