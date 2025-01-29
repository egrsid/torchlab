import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# TODO: НЕПРАВИЛЬНЫЙ ВЫВОД ТОЧНОСТИ
# TODO: Добавить verbose
# TODO: Добавить вывод графиков прямо во время обучения
# TODO: Добавить возможность автоматического подбора гиперпараметров (например, через Random Search или Grid Search)
# TODO: Добавить логирование результатов обучения с использованием TensorBoard
# TODO: Реализовать конфигурационный файл для задания гиперпараметров (например, config.yaml)
# TODO: Создать документацию для пользователя с примерами использования библиотеки
# TODO: Добавить опцию автоматического сохранения лучших весов модели на основе метрики валидации

class Trainer:
    def __init__(self, model, criterion, optimizer, train_loader, test_loader,
                 epochs=100, patience=5, scheduler=None, callbacks=None, average='binary',
                 device='cuda'):
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
        if callbacks is not None:
            self._history.update({'valid_' + call.__name__: [] for call in callbacks})
            self._history.update({'train_' + call.__name__: [] for call in callbacks})

        self._callbacks = callbacks
        self._average = average
        self._best_loss = float('inf')
        self._best_acc = 0

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

            if self._callbacks is not None:
                for call in self._callbacks:
                    self._history[f'train_{call.__name__}'].append(0)
                    self._history[f'valid_{call.__name__}'].append(0)


            train_loss, train_accuracy = self._train_one_epoch(epoch)
            valid_loss, valid_accuracy = self._validate_one_epoch()

            self._history['train_losses'].append(train_loss)
            self._history['train_accuracies'].append(train_accuracy)
            self._history['valid_losses'].append(valid_loss)
            self._history['valid_accuracies'].append(valid_accuracy)

            for call in self._callbacks or []:
                self._history['train_' + call.__name__][-1] /= len(self._train_loader)
                self._history['valid_' + call.__name__][-1] /= len(self._test_loader)

            if valid_loss < self._best_loss:
                self._best_loss = valid_loss
                self._best_acc = valid_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self._patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

            if self._scheduler:
                try:
                    self._scheduler.step()
                except Exception as e:
                    print(f'Error running scheduler: {e}')

            epoch_time = time.time() - start_time
            self._print_epoch_summary(epoch, epoch_time)

    def _train_one_epoch(self, epoch):
        self._model.train()
        running_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(self._train_loader, desc=f'Training Epoch {epoch + 1}/{self._epochs}', leave=False)
        for inputs, labels in pbar:
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

            current_loss = running_loss / (total / labels.size(0))
            current_accuracy = correct / total

            # Описание с текущими метриками
            mes = f'Training Epoch {epoch + 1}/{self._epochs} - Loss: {current_loss:.4f}, Accuracy: {current_accuracy:.4f}'
            if self._callbacks:
                for call in self._callbacks:
                    metric = call(predicted.to('cpu'), labels.to('cpu'), average=self._average)
                    self._history['train_' + call.__name__][-1] += metric
                    mes += f' {call.__name__.capitalize()}: {metric:.4f}'

            pbar.set_description(mes)

        return running_loss / len(self._train_loader), correct / total

    def _validate_one_epoch(self):
        self._model.eval()
        running_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            pbar = tqdm(self._test_loader, desc='Validating', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self._device), labels.to(self._device)
                outputs = self._model(inputs)
                loss = self._criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                for call in self._callbacks or []:
                    metric = call(predicted.to('cpu'), labels.to('cpu'), average=self._average)
                    self._history['valid_' + call.__name__][-1] += metric

        return running_loss / len(self._test_loader), correct / total

    def _print_epoch_summary(self, epoch, epoch_time):
        display = f'Epoch {epoch + 1}/{self._epochs} -'
            
        for name, metric in self._history.items():
            if name.startswith('train'):
                display += f" {' '.join(i.capitalize() for i in name.split('_'))}: {round(metric[-1], 3)}, "
                display += f" {' '.join(i.capitalize() for i in ['valid'] + name.split('_')[1:])}: {round(self._history['_'.join(['valid'] + name.split('_')[1:])][-1], 3)}, "
        display = f'{display[:-2]} - Time: {epoch_time:.2f}s'

        print(display)

    def plot_metrics(self):
        fig, axes = plt.subplots(1, len(list(self._history.keys())) // 2, figsize=(5 * len(list(self._history.keys())) // 2, 2 * len(self._callbacks) if self._callbacks is not None else 4))

        epochs = range(1, self._epochs+1)
        default_metrics = ['losses', 'accuracies']
        for ax, call in zip(axes, default_metrics + self._callbacks if self._callbacks is not None else default_metrics):
            name = call.__name__ if call not in default_metrics else call
            train_metric = self._history[f'train_{name}']
            valid_metric = self._history[f'valid_{name}']
            ax.plot(epochs, train_metric, label=f'train', color='blue')
            ax.plot(epochs, valid_metric, label=f'valid', color='orange')
            ax.set_title(name)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Value')
            ax.legend()
        
        plt.tight_layout()
        plt.show()