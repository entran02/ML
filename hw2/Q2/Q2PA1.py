import numpy as np
import pandas as pd

np.random.seed(7)

num_samples = 10000
class_priors = [0.3, 0.3, 0.4]

mean_class1 = np.array([2, 2, 2])
mean_class2 = np.array([4, 2, 2])
mean_class3a = np.array([3, 2+ np.sqrt(3), 2])
mean_class3b = np.array([3, (2+ np.sqrt(3))/2, 2 + np.sqrt(3)])

std_dev = 1.0

cov_matrix = np.identity(3)

labels = np.random.choice([1, 2, 3], size=num_samples, p=class_priors)

data = []

for label in labels:
    if label == 1:
        sample = np.random.multivariate_normal(mean_class1, cov_matrix * std_dev, 1)
    elif label == 2:
        sample = np.random.multivariate_normal(mean_class2, cov_matrix * std_dev, 1)
    else:
        choice = np.random.rand()
        mean_class3 = mean_class3a if choice < 0.5 else mean_class3b
        sample = np.random.multivariate_normal(mean_class3, cov_matrix * std_dev, 1)

    data.append(sample.flatten())

columns = ['x', 'y', 'z']
df = pd.DataFrame(data, columns=columns)

df['label'] = labels

df.to_csv('generated_data.csv', index=False)
