from ultralytics import YOLO
import os

# params
model_type = "yolo26n.pt"
model_path_output = "models/trained_models"
model_path = "../runs/detect/"
num_epochs = 25
batch_size =16

def train():
    print("starting training")
    model = YOLO(model_type)
    model.train(data="Annoted_Data/annoted_dataset_yolo/data.yaml", epochs=num_epochs, batch=batch_size, project=model_path_output, exist_ok=True)

    print("starting validation")
    model.val()

def test():
    model = YOLO(model_path + model_path_output + "/train/weights/best.pt")
    print("starting testing")
    results = model.predict("Test_data")

    scored_results = []
    for result in results:
        if len(result.boxes) > 0:
            max_conf = result.boxes.conf.max().item()
            scored_results.append((result, max_conf))

    scored_results.sort(key= lambda x: x[1], reverse=True)
    top_100 = scored_results[:100]

    for i, (result, score) in enumerate(top_100):
        result.save(filename=os.path.join("results", os.path.basename(result.path)))

    # results.sort(key= lambda r: 0 if r.probs is None else r.probs.top1conf[0].item(), reverse=True)
    # results = results[:100]
    # for result in results:
    #     print("None" if result.probs is None else result.probs.top1conf)
    #     result.save()

if __name__ == '__main__':
    if not os.path.exists(model_path + model_path_output):
        print("need to train model")
        train()
    test()