import os
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