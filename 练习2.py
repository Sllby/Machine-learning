import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

true_labels = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 1, 0]
])

predict_scores = np.array([
    [0.1, 0.2, 0.7],
    [0.1, 0.6, 0.3],
    [0.5, 0.2, 0.3],
    [0.1, 0.1, 0.8],
    [0.4, 0.2, 0.4],
    [0.6, 0.3, 0.1],
    [0.4, 0.2, 0.4],
    [0.4, 0.1, 0.5],
    [0.1, 0.1, 0.8],
    [0.1, 0.8, 0.1]
])


colors = ['red', 'blue', 'green']
labels = ['Class 0', 'Class 1', 'Class 2']
plt.figure(figsize=(10, 8))

for i in range(3):
    fpr, tpr, thresholds = metrics.roc_curve(true_labels[:, i], predict_scores[:, i])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i], label=f'{labels[i]} (AUC = {roc_auc:.2f})')


plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
