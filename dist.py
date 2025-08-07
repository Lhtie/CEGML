import numpy as np
import matplotlib.pyplot as plt

samples = np.clip(np.random.normal(loc=0.5, scale=0.1, size=10000), 0.0, 1.0)

plt.figure(figsize=(8, 4))
plt.hist(samples, bins=50, density=True, alpha=0.7, color='tab:blue', edgecolor='black')

plt.title('Truncated Normal Distribution (clipped N(0.5, 0.1^2) to [0, 1])')
plt.xlabel('Value')
plt.ylabel('Density')

plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()