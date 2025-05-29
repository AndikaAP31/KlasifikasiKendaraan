import cv2
import numpy as np
import matplotlib.pyplot as plt

# fungsi untuk membuat confusion matrix secar manual
def confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        raise ValueError("Panjang y_true dan y_pred harus sama.")
    
    classes = np.unique(np.concatenate((y_true, y_pred)))
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for t, p in zip(y_true, y_pred):
        i = label_to_index[t]
        j = label_to_index[p]
        cm[i, j] += 1
    
    return cm, classes

# fungsi untuk menghitung acccuracy score secara manual
def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("Panjang y_true dan y_pred harus sama.")
    return np.mean(y_true == y_pred)

# fungsi untuk menghitung dan membuat classification report secara manual
def classification_report(y_true, y_pred, target_names=None):
    cm, classes = confusion_matrix(y_true, y_pred)
    n_classes = len(classes)

    if target_names is None:
        target_names = [str(c) for c in classes]
    
    report = ""
    report += " " * 15 + "precision    recall  f1-score   support\n\n"
    
    precisions, recalls, f1s, supports = [], [], [], []

    for i, cls in enumerate(classes):
        TP = cm[i, i]
        FP = np.sum(cm[:, i]) - TP
        FN = np.sum(cm[i, :]) - TP
        support = np.sum(cm[i, :])

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        report += f"{target_names[i]:<15} {precision:>9.2f} {recall:>9.2f} {f1:>9.2f} {support:>9}\n"
    
    total_support = np.sum(supports)
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    weights = np.array(supports) / total_support
    weighted_precision = np.sum(weights * np.array(precisions))
    weighted_recall = np.sum(weights * np.array(recalls))
    weighted_f1 = np.sum(weights * np.array(f1s))

    report += "\n"
    report += f"{'accuracy':<44} {accuracy:>5.2f} {total_support:>9}\n"
    report += f"{'macro avg':<15} {macro_precision:>9.2f} {macro_recall:>9.2f} {macro_f1:>9.2f} {total_support:>9}\n"
    report += f"{'weighted avg':<15} {weighted_precision:>9.2f} {weighted_recall:>9.2f} {weighted_f1:>9.2f} {total_support:>9}\n"
    
    return report

# fungsi untuk menampilkan hasil evaluasi dari model SVM
def evaluate(y_true, y_pred, kernel_name, le):
    print(f"\n###Classification Report - {kernel_name}###")
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print(f"Accuracy Score: {accuracy_score(y_true, y_pred):.4f}")
    
    cm, classes = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {kernel_name}')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, le.classes_, rotation=45)
    plt.yticks(tick_marks, le.classes_)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color=color)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()

# fungsi untuk menampilkan visualisasi hasil prediksi dari model SVM
def visualize_predictions(y_true, y_pred, images, kernel_name, le, max_images=20):
    print(f"\n###Visualisasi Hasil Prediksi - {kernel_name}###")
    
    num_samples = min(len(y_true), max_images)
    cols = 5
    rows = (num_samples + cols - 1) // cols

    plt.figure(figsize=(15, rows * 2.5))
    for i in range(num_samples):
        plt.subplot(rows, cols, i + 1)
        true_label = le.inverse_transform([y_true[i]])[0]
        pred_label = le.inverse_transform([y_pred[i]])[0]

        color = 'green' if y_true[i] == y_pred[i] else 'red'
        plt.title(f"T: {true_label}\nP: {pred_label}", fontsize=9, color=color)
        plt.axis('off')

        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)

    plt.suptitle(f"Hasil Prediksi - {kernel_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()