import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader, epochs=100, patience=5, scheduler=None,
                 device='cuda'):
        # Приватные атрибуты
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._epochs = epochs
        self._scheduler = scheduler
        self._patience = patience
        self._device = device

        # История метрик
        self._history = {'train_losses': [], 'valid_losses': [], 'train_accuracies': [], 'valid_accuracies': []}
        self._best_loss = float('inf')
        self._best_acc = 0

    # Свойства для доступа к выбранным атрибутам
    @property
    def history(self):
        return self._history

    @property
    def model(self):
        return self._model

    @property
    def best_loss(self):
        return self._best_loss

    @property
    def best_acc(self):
        return self._best_acc

    @property
    def epochs(self):
        return self._epochs

    def train_model(self):
        self._model.to(self._device)
        epochs_without_improvement = 0

        for epoch in range(self._epochs):
            start_time = time.time()

            # Training phase
            self._model.train()
            running_loss, correct, total, num_batches = 0.0, 0, 0, 0

            pbar = tqdm(self._train_loader, desc=f'Training Epoch {epoch + 1}/{self._epochs}', leave=False)
            for inputs, labels in pbar:
                num_batches += 1
                inputs, labels = inputs.to(self._device), labels.to(self._device)

                self._optimizer.zero_grad()
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Обновляем текущий loss в tqdm
                current_loss = running_loss / num_batches
                current_accuracy = correct / total
                pbar.set_description(f'Training Epoch {epoch + 1}/{self._epochs} - '
                                     f'Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.4f}')

            train_loss = running_loss / num_batches
            train_accuracy = correct / total
            self._history['train_losses'].append(train_loss)
            self._history['train_accuracies'].append(train_accuracy)

            # Validation phase
            self._model.eval()
            running_loss, correct, total, num_batches = 0.0, 0, 0, 0

            with torch.no_grad():
                pbar = tqdm(self._test_loader, desc='Validating', leave=False)
                for inputs, labels in pbar:
                    num_batches += 1
                    inputs, labels = inputs.to(self._device), labels.to(self._device)
                    outputs = self._model(inputs)
                    loss = self._criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            valid_loss = running_loss / num_batches
            valid_accuracy = correct / total
            self._history['valid_losses'].append(valid_loss)
            self._history['valid_accuracies'].append(valid_accuracy)

            # Early stopping
            if valid_loss < self._best_loss:
                self._best_loss = valid_loss
                self._best_acc = valid_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self._patience:
                print(f'\nEarly stopping triggered after {epoch} epochs')
                break

            # Шаг scheduler
            if self._scheduler:
                try:
                    self._scheduler.step()
                except Exception as e:
                    print(f'Error running scheduler: {e}')

            end_time = time.time()
            epoch_time = end_time - start_time
            print(f'Epoch {epoch + 1}/{self._epochs} - '
                f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                f'Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f} - '
                f'Time: {epoch_time:.2f}s')



    def plot_metrics(self):
        train_losses = self._history['train_losses']
        valid_losses = self._history['valid_losses']
        train_accuracies = self._history['train_accuracies']
        valid_accuracies = self._history['valid_accuracies']

        epochs = range(1, len(train_losses) + 1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs, valid_losses, label='Valid Loss', color='orange')
        plt.title('Train and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, label='Train Accuracy', color='blue')
        plt.plot(epochs, valid_accuracies, label='Valid Accuracy', color='orange')
        plt.title('Train and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()