import matplotlib.pyplot as plt
import json
import os

def plot_training(history, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, 'training_plot.png'))
    plt.close()

def save_history(history, save_path):
    with open(os.path.join(save_path, 'history.json'), 'w') as f:
        json.dump(history.history, f)
