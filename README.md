# 🎙 Программный комплекс для детектирования синтетической речи

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **Выпускная квалификационная работа (Дипломный проект)**  
> **Студент:** Будревич Максим Игоревич  
> **Тема работы:** Обнаружение синтетической речи в системах биометрической аутентификации и цифровых коммуникациях

## 📖 Описание проекта

Данный проект является программным комплексом, предназначенным для обучения и использования моделей обнаружения синтетической (сгенерированной) речи (Anti-Spoofing). 

Приложение имеет модульную архитектуру и разделено на три основные части:
1. **Frontend:** Telegram-бот для удобного взаимодействия с пользователем.
2. **Backend:** REST API для обработки аудио и инференса моделей.
3. **Пространство для обучения:** Среда для подготовки датасетов, тренировки и тестирования моделей.

Такая структура позволяет легко заменять отдельные блоки при необходимости (например, подключить веб-интерфейс вместо бота), а также быстро обучать, тестировать и внедрять новые модели. Комплекс будет полезен исследователям и разработчикам, чья цель — спроектировать надежную антиспуфинг-систему.

## 🛠 Стек технологий

- **Frontend:** Python, `aiogram` (Telegram Bot API)
- **Backend:** Python, `FastAPI`, `Uvicorn`
- **Machine Learning & Обучение:** `PyTorch`, `PyTorch Lightning`, `MLflow` (для трекинга экспериментов)

## 🚀 Установка и локальный запуск

### 1. Подготовка окружения
Склонируйте репозиторий и перейдите в папку проекта:
```bash
git clone https://github.com/BudrevichMaksim/diplom.git
cd diplom
```

Рекомендуется создать виртуальное окружение и установить зависимости:
```bash
python -m venv venv
source venv/bin/activate  # Для Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Обучение модели (опционально)
Если вам нужно обучить модель с нуля:
1. Откройте Jupyter Notebook `testing_machine.ipynb`.
2. Загрузите и подготовьте датасет (структура директорий описана внутри ноутбука, также имеется скрипт для загрузки asvspoof5).
3. Убедитесь, что у вас установлен и запущен `MLflow` для логирования метрик.
4. Запустите ячейки для тренировки.

### 3. Настройка переменных окружения
Создайте файл `.env` в корне проекта (или переименуйте `.env.example`) и заполните его по следующему шаблону:

```env
BOT_TOKEN="ваш_токен_телеграм_бота"
API_PATH="http://127.0.0.1:8000"
EXTRACTOR="тип_экстрактора" # Например: mel (как описано в API)
DETECTOR="тип_детектора"    # Например: rcnn, wavlm (как описано в API)
```

### 4. Запуск комплекса

Для работы системы необходимо запустить Backend-сервер и Telegram-бота. Откройте два разных терминала.

**Терминал 1: Запуск API (FastAPI)**
```bash
uvicorn api.main:app
# или с флагом --reload для режима разработки
uvicorn api.main:app --reload
```

**Терминал 2: Запуск Telegram-бота**
```bash
python -m bot.main
```

## 🗂 Структура проекта (кратко)
```text
diplom/
├── api/                  # Исходный код FastAPI (эндпоинты, обработка аудио)
├── bot/                  # Исходный код Telegram-бота на aiogram
├── ml/             			# Cкрипты для ML
├── requirements.txt      # Список зависимостей Python
├── testing_machine.ipynb # Ноутбук для обучения
└── .env                  # Конфигурационный файл (не коммитится в Git)
```

# 🎙 Synthetic Speech Detection Software Suite

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> **Graduation Thesis (Degree Project)**  
> **Student:** Maksim I. Budrevich  
> **Thesis Topic:** Synthetic speech detection in biometric authentication systems and digital communications

## 📖 Project Description

This project is a software suite designed for training and deploying synthetic (generated) speech detection models (Anti-Spoofing). 

The application features a modular architecture and is divided into three main components:
1. **Frontend:** A Telegram bot for convenient user interaction.
2. **Backend:** A REST API for audio processing and model inference.
3. **Training Environment:** A workspace for dataset preparation, model training, and testing.

This structure allows for easy replacement of individual modules if necessary (e.g., connecting a web interface instead of the bot), as well as rapid training, testing, and deployment of new models. The suite will be highly beneficial for researchers and developers aiming to design a reliable anti-spoofing system.

## 🛠 Tech Stack

- **Frontend:** Python, `aiogram` (Telegram Bot API)
- **Backend:** Python, `FastAPI`, `Uvicorn`
- **Machine Learning & Training:** `PyTorch`, `PyTorch Lightning`, `MLflow` (for experiment tracking)

## 🚀 Installation and Local Setup

### 1. Environment Setup
Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/BudrevichMaksim/diplom.git
cd diplom
```

It is recommended to create a virtual environment and install the dependencies:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Model Training (Optional)
If you need to train a model from scratch:
1. Open the `testing_machine.ipynb` Jupyter Notebook.
2. Download and prepare the dataset (the directory structure is described inside the notebook; a script for downloading ASVspoof 5 is also provided).
3. Ensure that `MLflow` is installed and running for metric logging.
4. Run the cells to start training.

### 3. Environment Variables Configuration
Create an `.env` file in the project root (or rename `.env.example`) and fill it out according to the following template:

```env
BOT_TOKEN="your_telegram_bot_token"
API_PATH="http://127.0.0.1:8000"
EXTRACTOR="extractor_type" # Example: mel (as described in the API)
DETECTOR="detector_type"   # Example: rcnn, wavlm (as described in the API)
```

### 4. Running the Application

To run the system, you need to start the Backend server and the Telegram bot. Open two separate terminals.

**Terminal 1: Start the API (FastAPI)**
```bash
uvicorn api.main:app
# or with the --reload flag for development mode
uvicorn api.main:app --reload
```

**Terminal 2: Start the Telegram Bot**
```bash
python -m bot.main
```

## 🗂 Project Structure (Brief)
```text
diplom/
├── api/                  # FastAPI source code (endpoints, audio processing)
├── bot/                  # Telegram bot source code (aiogram)
├── ml/                   # Machine Learning scripts
├── requirements.txt      # Python dependencies list
├── testing_machine.ipynb # Notebook for training models
└── .env                  # Configuration file (ignored by Git)
```

## 📬 Контакты / Contact

[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:maksimbudrevich@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/BudrevichMaksim)
