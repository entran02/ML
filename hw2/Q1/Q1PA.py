import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

m01 = [3, 0]
C01 = [[2, 0], [0, 1]]
m02 = [0, 3]
C02 = [[1, 0], [0, 2]]
m1 = [2, 2]
C1 = [[1, 0], [0, 1]]
w1 = 0.65
w2 = 0.35

data_samples = pd.read_csv("data_samples.csv")
class0_samples = data_samples[data_samples['label'] == 0]
class1_samples = data_samples[data_samples['label'] == 1]

threshold_vals = np.concatenate([
    np.array([0, 0.01, 0.1]),
    np.arange(1, 100, .5),
    np.arange(100, 1000, 100),
    np.array([10000, 100000, float("inf")])
])

def classify_from_threshold(samples, threshold):
    L01 = multivariate_normal.pdf(samples, mean=m01, cov=C01)
    L02 = multivariate_normal.pdf(samples, mean=m02, cov=C02)
    L0 = (L01 + L02) / 2
    L1 = multivariate_normal.pdf(samples, mean=m1, cov=C1)
    return (L1 / L0) > threshold

class0_results = {threshold: classify_from_threshold(class0_samples[["x1", "x2"]].to_numpy(), threshold) for threshold in threshold_vals}
class1_results = {threshold: classify_from_threshold(class1_samples[["x1", "x2"]].to_numpy(), threshold) for threshold in threshold_vals}

tp_rate = [np.mean(class1_results[threshold]) for threshold in threshold_vals]
fp_rate = [np.mean(class0_results[threshold]) for threshold in threshold_vals]

diag_dist = np.sqrt(np.array(fp_rate) ** 2 + (np.array(tp_rate) - 1) ** 2)
optimal_idx = np.argmin(diag_dist)
theoretical_error_min = (1 - tp_rate[optimal_idx]) * w2 + fp_rate[optimal_idx] * w1

errors = np.array(fp_rate) * w1 + (1 - np.array(tp_rate)) * w2
empirical_optimal_idx = np.argmin(errors)

plt.figure(figsize=(12, 6))
plt.plot(fp_rate, tp_rate, linestyle='-', color='orange', label='ROC Curve')
plt.plot([0, 1], [0, 1], linestyle='--', color='r', label='y = x')
plt.scatter(fp_rate[optimal_idx], tp_rate[optimal_idx], color="red", marker='o', s=100, label=f'Theoretical optimal threshold={threshold_vals[optimal_idx]:.2f}, P(error)={theoretical_error_min:.2f}')
plt.scatter(fp_rate[empirical_optimal_idx], tp_rate[empirical_optimal_idx], color="green", marker='o', s=100, label=f'Empirical optimal threshold={threshold_vals[empirical_optimal_idx]:.2f}, P(error)={errors[empirical_optimal_idx]:.2f}')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()