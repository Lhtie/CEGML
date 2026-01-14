import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl

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
    plt.title('Accuracy against #TrainingRound Curve', fontsize=16)
    plt.xlabel('#TrainingRounds', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(num_samples[-1], color='red', linestyle='--', label="Total Train Round", linewidth=1)
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

def plot_3dSurface(Z):
    dfa_sizes = np.arange(3, 11)      # 3..10 (8)
    star_depths = np.arange(0, 5)     # 0..4  (5)

    Y, X = np.meshgrid(dfa_sizes, star_depths)  # X: DFA size, Y: star depth

    Z_center = np.median(Z, axis=2)                 # or Z.mean(axis=2)
    Z_min = Z.min(axis=2)
    Z_max = Z.max(axis=2)

    mpl.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "figure.dpi": 120,       # interactive
        "savefig.dpi": 300,      # export
    })

    fig = plt.figure(figsize=(7.8, 5.6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    # Colormap: low=green, high=red
    cmap = plt.colormaps.get_cmap("RdYlGn_r")
    norm = Normalize(vmin=Z_center.min(), vmax=Z_center.max())

    ax.plot_surface(
        X, Y, Z_center,
        facecolors=cmap(norm(Z_center)),
        rstride=1, cstride=1,
        linewidth=0.25,
        antialiased=True,
        shade=False
    )

    for i in range(Y.shape[0]):
        for j in range(X.shape[1]):    
            ax.plot(
                [X[i, j], X[i, j]],
                [Y[i, j], Y[i, j]],
                [Z_min[i, j], Z_max[i, j]],
                linewidth=1.0,
                alpha=0.75
            )

    # Colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(Z_center)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.08)
    cbar.set_label("Samples needed (lower = easier)")

    ax.set_xlabel("Star depth")
    ax.set_ylabel("Minimal DFA size (states)")
    ax.set_zlabel("Sample complexity")

    # Ticks: keep sparse & readable
    ax.set_xticks(star_depths)
    ax.set_yticks(dfa_sizes)

    # View: choose a stable, readable angle
    ax.view_init(elev=24, azim=-55)
    ax.set_box_aspect((1.25, 1.0, 0.7))

    # Optional: remove pane fills for cleaner look
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.savefig("accuracy_curves/surface_paper.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    Z = np.array([
        [[ 7.52,  8.00,  8.56], [13.16, 14.00, 14.98], [18.80, 20.00, 21.40], [24.44, 26.00, 27.82],
        [30.08, 32.00, 34.24], [35.72, 38.00, 40.66], [41.36, 44.00, 47.08], [47.00, 50.00, 53.50]],

        [[ 9.00,  9.00,  9.00], [33.33, 33.33, 33.33], [16.33, 16.33, 16.33], [58.00, 58.00, 58.00],
        [44.00, 44.00, 44.00], [12.00, 27.33, 12.00], [55.27, 58.80, 62.92], [61.37, 65.30, 69.87]],

        [[26.32, 28.00, 29.96], [28.33, 28.33, 28.33], [35.72, 38.00, 40.66], [42.07, 44.75, 47.90],
        [48.41, 51.50, 55.11], [69.00, 69.00, 69.00], [61.10, 65.00, 69.55], [67.44, 71.75, 76.73]],

        [[34.32, 36.50, 39.06], [41.23, 43.85, 46.91], [48.13, 51.20, 54.78], [55.04, 58.55, 62.65],
        [61.94, 65.90, 70.53], [68.85, 73.25, 78.41], [75.75, 80.60, 86.24], [82.66, 87.95, 94.16]],

        [[45.12, 48.00, 51.36], [53.77, 57.20, 61.20], [62.42, 66.40, 71.05], [71.06, 75.60, 80.89],
        [79.71, 84.80, 90.74], [88.36, 94.00,100.58], [97.01,103.20,110.42], [105.66,112.40,120.27]]
    ], dtype=float)
    # Z = np.array([
    #     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #     [[9, 9, 9], [33.33, 33.33, 33.33], [16.33, 16.33, 16.33], [58, 58, 58], [44, 44, 44], [12, 27.33, 12], [0, 0, 0], [0, 0, 0]],
    #     [[0, 0, 0], [28.33, 28.33, 28.33], [0, 0, 0], [0, 0, 0], [0, 0, 0], [69, 69, 69], [0, 0, 0], [0, 0, 0]],
    #     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
    #     [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # ])
    plot_3dSurface(Z)
    exit(0)

    # names = ["baseline", "50_50_CEs", "100_CEs", "random_CEs", "normal_CEs"]
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    names = ["", "_ce"]
    colors = ['tab:blue', 'tab:red']
    # names = ["posrate=0.05", "posrate=0.1", "posrate=0.23", "posrate=0.3", "posrate=0.5", "posrate=0.8"]
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    # names = ["num_aug=1-aug_strategy=dfa_state", "num_aug=5-aug_strategy=dfa_state", "num_aug=5-aug_strategy=random", "num_aug=5-aug_strategy=repeat"]
    # colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple']

    data = {}
    for name in names:
        # with open(os.path.join("accuracy_curves", f"Regex=(a(a)*b)*-mode={name}-train_length=8-test_length=8-num_aug=1-aug_strategy=dfa_state-epochs_per_round=3.json"), "r") as f:
        with open(os.path.join("accuracy_curves", f"icl_model=gpt5_totTrain=384_startSize=3_scaleFactor=2.0_totEval=32_evalBatch=32{name}.json"), "r") as f:
            data[name] = json.load(f)

    plt.figure(figsize=(10, 6))
    for (k, d), c in zip(data.items(), colors):
        num_samples = d["num_samples"]
        accs = d["accs"]
        smoothed_accs = smooth(accs, window_size=3)

        plt.plot(num_samples, smoothed_accs, label=k, color=c, linewidth=2)
        
    plt.title('Smoothed Accuracy Curve', fontsize=16)
    plt.xlabel('#TrainingSamples', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join("accuracy_curves", "ICL_model=gpt5_totTrain=384_CE.png"), dpi=500)