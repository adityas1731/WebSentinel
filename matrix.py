from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Assuming you have predicted probabilities and true labels
# Replace 'y_true' with your true labels and 'y_scores' with your predicted probabilities
fpr, tpr, thresholds = roc_curve([0,1], [0,1])
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
