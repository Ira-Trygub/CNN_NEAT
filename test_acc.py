from sklearn.metrics import accuracy_score
import torch


def test_acc(test_loader, model):
   
    model.eval()  # Устанавливаем режим оценки для модели
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Отключаем вычисление градиентов для оценки
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)  # Предсказанные метки
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy
