import numpy as np
import pandas as pd

NUM_SAMPLES = 10000
PRIOR_L0 = 0.65
PRIOR_L1 = 0.35

m01 = np.array([3, 0])
C01 = np.array([[2, 0], [0, 1]])

m02 = np.array([0, 3])
C02 = np.array([[1, 0], [0, 2]])

m1 = np.array([2, 2])
C1 = np.array([[1, 0], [0, 1]])

labels = []
samples = []

for _ in range(NUM_SAMPLES):
    label = 0 if np.random.uniform(0, 1) < PRIOR_L0 else 1
    labels.append(label)

    if label == 0:
        gaussian_choice = np.random.choice([1, 2], p=[0.5, 0.5])
        if gaussian_choice == 1:
            sample = np.random.multivariate_normal(m01, C01)
        else:
            sample = np.random.multivariate_normal(m02, C02)
    else:
        sample = np.random.multivariate_normal(m1, C1)
    samples.append(sample)

df = pd.DataFrame(samples, columns=["x1", "x2"])
df["label"] = labels

df.to_csv("data_samples.csv", index=False)
