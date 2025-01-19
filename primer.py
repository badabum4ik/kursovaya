

# Обучение модели


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# Архитектура сети (с динамическим вычислением размеров)
class FaceRecognitionCNN(nn.Module):
    def __init__(self):
        super(FaceRecognitionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Размер будет пересчитан позже
        self.flatten_size = None
        self.fc1 = nn.Linear(1, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))

        # Вычисляем размер только один раз
        if self.flatten_size is None:
            self.flatten_size = x.view(x.size(0), -1).shape[1]
            self.fc1 = nn.Linear(self.flatten_size, 128)

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Подготовка данных
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Использование датасета для обучения
train_dataset = datasets.ImageFolder("C:/Users/Илья/PycharmProject/kursovaya/train_data", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Инициализация модели, функции потерь и оптимизатора
model = FaceRecognitionCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Цикл обучения
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Эпоха {epoch + 1}/{num_epochs}, Потери: {running_loss:.4f}")


# Тестирование


# Перевод модели в режим оценки
model.eval()
test_dataset = datasets.ImageFolder("C:/Users/Илья/PycharmProject/kursovaya/test_data", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

correct = 0
total = 0
with torch.no_grad():  # Отключение вычисления градиентов
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Точность модели: {accuracy:.2f}%")


# Код для предобработки изображений и предсказаний


import cv2  # Для обработки изображений
import numpy as np

# Функция для предсказания с использованием OpenCV
def preprocess_image_with_opencv(image_path):
    # Чтение изображения с помощью OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Изображение по пути {image_path} не найдено.")

    # Преобразование изображения в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Изменение размера изображения
    image = cv2.resize(image, (128, 128))
    # Нормализация и преобразование в тензор
    image = np.transpose(image, (2, 0, 1))  # Меняем оси с (H, W, C) на (C, H, W)
    image = image / 255.0  # Приводим значения пикселей в диапазон [0, 1]
    image = (image - 0.5) / 0.5  # Нормализация как в обучении
    return torch.tensor(image, dtype=torch.float32)

def predict(image_path, model):
    # Предобработка изображения
    image = preprocess_image_with_opencv(image_path).unsqueeze(0)  # Добавление batch-измерения

    # Прогнозирование
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# Тестирование предсказания
result = predict("face.jpg", model)
print(f"Распознанный класс: {result}")

