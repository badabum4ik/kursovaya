

# API на Flask


from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
import pymysql
import json
from primer import preprocess_image_with_opencv  # Импорт функции обработки изображения

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Функция для сохранения результатов в базу данных MySQL
def save_to_db(image_path, result):
    connection = None
    cursor = None

    try:
        # Устанавливаем соединение с MySQL через pymysql
        connection = pymysql.connect(
            host="localhost",
            user="root",
            password="root",
            database="image_analysis"  # Название базы данных
        )

        cursor = connection.cursor()  # Курсор для выполнения SQL-запросов

        # Если результат - массив, преобразуем его в JSON-объект с ключами
        if isinstance(result, list):
            result = {"data": result}  # Упакуем массив в объект с ключом "data"

        # Преобразуем результат в строку JSON
        result_json = json.dumps(result, ensure_ascii=False)

        # SQL-запрос для вставки данных в таблицу
        query = """
        INSERT INTO analysis_results (image_path, result)
        VALUES (%s, %s);
        """

        # Выполняем вставку данных в таблицу
        cursor.execute(query, (image_path, result_json))

        # Подтверждаем изменения
        connection.commit()

        print("Данные успешно сохранены в базу данных.")

    except pymysql.MySQLError as err:
        print(f"Ошибка при подключении или работе с базой данных: {err}")
    finally:
        # Закрытие курсора и соединения
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# Маршрут для загрузки и обработки изображения
@app.route('/upload', methods=['POST'])
def upload_image():
    print("Запрос получен")  # Логирование

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Сохраняем загруженный файл
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    print(f"Файл сохранен: {filepath}")  # Логирование

    try:
        # Обработка изображения
        result = preprocess_image_with_opencv(filepath)

        # Преобразуем сложные типы данных (например, тензоры) в читаемый формат
        if isinstance(result, torch.Tensor):
            result = result.tolist()  # Преобразуем тензор в список
        elif isinstance(result, np.ndarray):
            result = result.tolist()  # Преобразуем массив NumPy в список

        # Пример структуры данных для сохранения в БД
        processed_result = {
            "description": "Результат обработки изображения",
            "data": result
        }

        # Сохраняем результат в базу данных
        save_to_db(filepath, processed_result)

        return jsonify({'result': processed_result}), 200
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

# Главная страница
@app.route('/')
def index():
    return jsonify({'message': 'Сервер работает!'})

# Запуск сервера
if __name__ == '__main__':
    app.run(debug=True)


# запрос в терминал
# curl.exe -X POST -F "file=@C:/Users/Илья/PycharmProject/kursovaya/test_data/class_2/image4.jpg" http://127.0.0.1:5000/upload


