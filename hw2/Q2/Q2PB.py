import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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

loss_10 = np.array([[0, 1, 10], [1, 0, 10], [1, 1, 0]])
loss_100 = np.array([[0, 1, 100], [1, 0, 100], [1, 1, 0]])

def expected_loss(sample, loss_matrix):
    posteriors = [
        dist_class1.pdf(sample) * class_priors[0],
        dist_class2.pdf(sample) * class_priors[1],
        0.5 * (dist_class3a.pdf(sample) + dist_class3b.pdf(sample)) * class_priors[2]
    ]
    total_prob = sum(posteriors)
    posteriors = [p/total_prob for p in posteriors]

    expected_losses = [sum([posteriors[j] * loss_matrix[i, j] for j in range(3)]) for i in range(3)]
    return np.argmin(expected_losses) + 1

df['predicted_label_10'] = df.apply(lambda row: expected_loss(row[['x', 'y', 'z']].values, loss_10), axis=1)
df['predicted_label_100'] = df.apply(lambda row: expected_loss(row[['x', 'y', 'z']].values, loss_100), axis=1)

confusion_10 = pd.crosstab(df['label'], df['predicted_label_10'], normalize='index')
confusion_100 = pd.crosstab(df['label'], df['predicted_label_100'], normalize='index')
print("Confusion Matrix for Λ10:")
print(confusion_10)
print("\nConfusion Matrix for Λ100:")
print(confusion_100)

def compute_expected_risk(confusion, loss_matrix):
    """
    Compute the expected risk using the confusion matrix and the loss matrix.
    """
    C = len(confusion)
    risk = 0
    for i in range(C):
        for j in range(C):
            risk += loss_matrix[i, j] * confusion.iloc[i, j]
    return risk

risk_10 = compute_expected_risk(confusion_10, loss_10)
risk_100 = compute_expected_risk(confusion_100, loss_100)

print("\nExpected Risk for Λ10:")
print(risk_10)

print("\nExpected Risk for Λ100:")
print(risk_100)

fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

colors = {'1': 'blue', '2': 'green', '3': 'red'}
class_labels = ['Class 1', 'Class 2', 'Class 3']

for label, color in colors.items():
    indices = df[df['predicted_label_10'] == int(label)].index
    ax1.scatter(df.loc[indices, 'x'], df.loc[indices, 'y'], df.loc[indices, 'z'], color=color, label=f'Class {label}')
ax1.set_title('Classification with Λ10')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.legend()

for label, color in colors.items():
    indices = df[df['predicted_label_100'] == int(label)].index
    ax2.scatter(df.loc[indices, 'x'], df.loc[indices, 'y'], df.loc[indices, 'z'], color=color, label=f'Class {label}')
ax2.set_title('Classification with Λ100')
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Feature 3')
ax2.legend()

plt.tight_layout()
plt.show()


