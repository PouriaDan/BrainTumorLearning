import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def set_ax_borders(ax, bottom=True, left=True, top=False, right=False):
    ax.spines['top'].set_visible(top)
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)

def plot_cm(gts, preds, ax=None, normalize=True):
    cm = confusion_matrix(gts, preds)
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        cm_normalized = cm
    annotations = np.array([
        [f"{cm_normalized[i, j]:.1%}\n({cm[i, j]})" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])
    sns.heatmap(cm_normalized, annot=annotations, fmt='', ax=ax, 
                cmap="Blues_r", vmin=0, vmax=1 if normalize else None)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return cm

def plot_class_distribution(train_labels, val_labels, test_labels, ax=None):
    train_counts = np.unique(train_labels, return_counts=True)[1]
    test_counts = np.unique(test_labels, return_counts=True)[1]
    val_counts = np.unique(val_labels, return_counts=True)[1]
    total_classes = len(train_counts)
    x = np.arange(total_classes)
    width = 0.3
    ax.bar(x - width, train_counts, width, label='Train')
    ax.bar(x, val_counts, width, label='Validation')
    ax.bar(x + width, test_counts, width, label='Test')
    ax.set_xticks(x)
    ax.set_xticklabels(np.arange(total_classes))
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.legend()
    set_ax_borders(ax)
    return train_counts, test_counts, val_counts
