import os
import json
import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curve(losses, outdir, name):
    epochs = len(losses)
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), losses, label='Loss', color='blue', linewidth=2)
    plt.title('Gradient Descent Loss Curve', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(epochs-1, color='red', linestyle='--', label="Final Epoch", linewidth=1)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=500)

def plot_accuracy_curve(num_samples, accs, outdir, name):
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump({"num_samples": num_samples, "accs": accs}, f)

    plt.figure(figsize=(10, 6))
    plt.plot(num_samples, accs, label='Accuracy', color='blue', linewidth=2)
    plt.title('Accuracy against #TrainingSamples Curve', fontsize=16)
    plt.xlabel('#TrainingSamples', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(num_samples[-1], color='red', linestyle='--', label="Total Train Num", linewidth=1)
    plt.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, f"{name}.png"), dpi=500)

def smooth(y, window_size=5):
    if window_size < 2:
        return y.copy()
    pad = window_size // 2
    y_padded = np.pad(y, pad_width=pad, mode='reflect')
    window = np.ones(window_size) / window_size
    return np.convolve(y_padded, window, mode='valid')

if __name__ == "__main__":
    # names = ["baseline", "50_50_CEs", "100_CEs", "random_CEs", "normal_CEs"]
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    names = ["baseline", "random_CEs"]
    colors = ['tab:blue', 'tab:red']

    data = {}
    for name in names:
        with open(os.path.join("accuracy_curves", f"Overall_Accuracy_{name}.json"), "r") as f:
            data[name] = json.load(f)

    plt.figure(figsize=(10, 6))
    for (k, d), c in zip(data.items(), colors):
        num_samples = d["num_samples"]
        accs = d["accs"]
        smoothed_accs = smooth(accs, window_size=7)

        plt.plot(num_samples, smoothed_accs, label=k, color=c, linewidth=2)
        
    plt.title('Smoothed Accuracy Curve', fontsize=16)
    plt.xlabel('#TrainingSamples', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join("accuracy_curves", "Overall_Accuracy_Smoothed.png"), dpi=500)