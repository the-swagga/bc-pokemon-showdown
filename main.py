from src.data_preprocessing.encode_and_normalise_data import preprocess_data
from src.model.train import train_model
from src.model.quant_eval import quantitative_evaluate_model

# This main.py file is for model training and evaluation; for data collection, go to /src/data_collection/poke_env_main.py

def preprocess_train_evaluate(seed=38, records=0, show_loss=True):
    preprocess_data(seed=seed, records=records)
    train_model(seed=seed, show_loss=show_loss)
    acc = quantitative_evaluate_model()
    print()
    return acc

acc = 0
top3acc = 0
cum_acc = 0
cum_top3acc = 0
best_acc = -1
best_seed = -1

#for i in range(1, 1001):
    #acc, top3acc = preprocess_train_evaluate(seed=i, records=1776, show_loss=False)
    #cum_acc += acc
    #cum_top3acc += top3acc

    #if acc > best_acc:
        #best_acc = acc
        #best_seed = i

#print(f"Average Accuracy: {cum_acc/1000}")
#print(f"Average Top 3 Accuracy: {cum_top3acc/1000}")
#print(f"Best Seed by Average Accuracy: {best_seed}")

preprocess_train_evaluate(seed=173, records=1776, show_loss=True)