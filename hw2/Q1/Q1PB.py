import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_samples = pd.read_csv("data_samples.csv")
samples = data_samples[["x1", "x2"]].to_numpy()
labels = data_samples["label"].to_numpy()

class0_samples = samples[labels == 0]
class1_samples = samples[labels == 1]

mean0 = np.mean(class0_samples, axis=0)
mean1 = np.mean(class1_samples, axis=0)

S0 = np.cov(class0_samples, rowvar=False, bias=True) * len(class0_samples)
S1 = np.cov(class1_samples, rowvar=False, bias=True) * len(class1_samples)

Sw = S0 + S1

mean_diff = (mean1 - mean0).reshape(-1, 1)
Sb = len(class0_samples) * len(class1_samples) * mean_diff.dot(mean_diff.T) / len(samples)

eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw).dot(Sb))

W_LDA = eigvecs[:, np.argmax(eigvals)]

projection_class0 = class0_samples.dot(W_LDA)
projection_class1 = class1_samples.dot(W_LDA)

thresholds = np.linspace(min(projection_class0), max(projection_class1), 400)
tp_rates, fp_rates = [], []

errors = []
for threshold in thresholds:
    fp = np.sum(projection_class0 > threshold) / len(class0_samples)
    fn = np.sum(projection_class1 < threshold) / len(class1_samples)
    tp = 1 - fn
    
    fp_rates.append(fp)
    tp_rates.append(tp)
    
    error = 0.5 * (fp + fn)
    errors.append(error)

min_error_idx = np.argmin(errors)
optimal_threshold = thresholds[min_error_idx]

plt.figure(figsize=(12, 6))
plt.plot(fp_rates, tp_rates, label='LDA ROC Curve')
plt.scatter(fp_rates[min_error_idx], tp_rates[min_error_idx], color='red', s=100, zorder=5, label=f"Min P(error) @ Threshold={optimal_threshold:.2f}")
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('LDA ROC Curve')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


