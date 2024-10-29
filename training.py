import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, epochs, train_loader, test_loader, patience=5, device='cpu'):
    # Переменные для сохранения метрик
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    # Переменная для ранней остановки
    best_loss = float('inf')
    epochs_without_improvement = 0

    model.to(device)

    for epoch in range(epochs):
        start_time = time.time()

        model.train()  # Устанавливаем модель в режим тренировки
        running_loss = 0.0
        correct = 0
        total = 0

        # Обучение
        pbar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Обнуляем градиенты
            outputs = model(inputs)  # Прямой проход
            loss = criterion(outputs, labels)  # Вычисляем потери
            loss.backward()  # Обратное распространение
            optimizer.step()  # Обновляем параметры

            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Вывод метрик с помощью tqdm
            current_loss = running_loss / (len(train_loader) + 1e-10)
            current_accuracy = correct / (total + 1e-10)
            pbar.set_description(f'Training Epoch {epoch + 1}/{epochs} - '
                                 f'Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.4f}')

        # Расчет метрик для обучающего набора
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Валидация
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Validating', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Расчет метрик для валидационного набора
        valid_loss = running_loss / len(test_loader)
        valid_accuracy = correct / total
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

        # Раннее прекращение обучения
        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f'\nEarly stopping triggered after {epoch + 1} epochs')
            break

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch + 1}/{epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
              f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f} - '
              f'Time: {epoch_time:.2f}s')

    return model, {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'train_accuracies': train_accuracies,
        'valid_accuracies': valid_accuracies
    }


def plot_metrics(metrics):
    # Извлечение данных
    train_losses = metrics['train_losses']
    valid_losses = metrics['valid_losses']
    train_accuracies = metrics['train_accuracies']
    valid_accuracies = metrics['valid_accuracies']

    epochs = range(1, len(train_losses) + 1)

    # plot для потерь
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='blue')
    plt.plot(epochs, valid_losses, label='Valid Loss', color='orange')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # pllot для точности
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(epochs, valid_accuracies, label='Valid Accuracy', color='orange')
    plt.title('Train and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
