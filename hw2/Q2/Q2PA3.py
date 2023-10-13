import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

df['predicted_label'] = df.apply(lambda row: classify_sample(row[['x', 'y', 'z']].values), axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

markers = {1: 'o', 2: '^', 3: 's'}
legend_handles = []

for label, marker in markers.items():
    subset = df[df['label'] == label]
    correct_subset = subset[subset['label'] == subset['predicted_label']]
    incorrect_subset = subset[subset['label'] != subset['predicted_label']]
    
    if not correct_subset.empty:
        handle = ax.scatter(correct_subset['x'], correct_subset['y'], correct_subset['z'], marker=marker, c='g', label=f"Class {label} (Correct)")
        legend_handles.append(handle)
    if not incorrect_subset.empty:
        handle = ax.scatter(incorrect_subset['x'], incorrect_subset['y'], incorrect_subset['z'], marker=marker, c='r', label=f"Class {label} (Incorrect)")
        legend_handles.append(handle)

ax.legend(handles=legend_handles, loc='best')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.title('3D Scatter Plot of Samples')
plt.show()