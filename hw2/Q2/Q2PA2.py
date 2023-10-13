import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

df = pd.read_csv('generated_data.csv')

mean_class1 = np.array([2, 2, 2])
mean_class2 = np.array([4, 2, 2])
mean_class3a = np.array([3, 2+ np.sqrt(3), 2])
mean_class3b = np.array([3, (2+ np.sqrt(3))/2, 2+ np.sqrt(3)])
cov_matrix = np.identity(3)

class_priors = [0.3, 0.3, 0.4]

dist_class1 = multivariate_normal(mean_class1, cov_matrix)
dist_class2 = multivariate_normal(mean_class2, cov_matrix)
dist_class3a = multivariate_normal(mean_class3a, cov_matrix)
dist_class3b = multivariate_normal(mean_class3b, cov_matrix)

def classify_sample(sample):
    likelihoods = [
        dist_class1.pdf(sample) * class_priors[0],
        dist_class2.pdf(sample) * class_priors[1],
        0.5 * (dist_class3a.pdf(sample) + dist_class3b.pdf(sample)) * class_priors[2]
    ]
    return np.argmax(likelihoods) + 1

df['predicted_label'] = df.apply(lambda row: classify_sample(row[['x', 'y', 'z']]), axis=1)

confusion_matrix = pd.crosstab(df['label'], df['predicted_label'], normalize='index')

print("Confusion Matrix:")
print(confusion_matrix)
