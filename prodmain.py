from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QComboBox
from PyQt6.QtCore import QTimer
import sys
import sounddevice as sd
import soundfile as sf
from threading import Thread
from datetime import datetime
import os
import requests
import uuid
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

def resource_path(relative_path):
    """
    Возвращает абсолютный путь к ресурсу, корректно работает как в режиме разработки,
    так и при запуске упакованного приложения (PyInstaller).
    """
    try:
        # При запуске из PyInstaller временная папка сохраняется в sys._MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class TranscriptionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Заголовок окна
        self.setWindowTitle("Transcription App")

        # Макет
        layout = QVBoxLayout()

        # Поле для отображения текста
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        layout.addWidget(self.text_display)

        # Выпадающий список для выбора устройства ввода
        self.device_selector = QComboBox()
        self.populate_device_list()
        layout.addWidget(self.device_selector)

        # Поле для ввода длительности чанка
        self.duration_input = QLineEdit()
        self.duration_input.setText("20")  # Значение по умолчанию
        self.duration_input.setPlaceholderText("Введите длительность чанка (в секундах)")
        layout.addWidget(self.duration_input)

        # Кнопка для начала записи
        self.start_button = QPushButton("Начать запись")
        self.start_button.clicked.connect(self.start_recording)
        layout.addWidget(self.start_button)

        # Кнопка для остановки записи
        self.stop_button = QPushButton("Остановить запись")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_recording)
        layout.addWidget(self.stop_button)

        # Кнопка для копирования текста в буфер обмена
        self.copy_button = QPushButton("Скопировать в буфер обмена")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        layout.addWidget(self.copy_button)

        # Применяем макет
        self.setLayout(layout)

        # Параметры записи
        self.fs = 16000  # Частота дискретизации
        self.channels = 1  # Моно запись
        self.recording = False

        # Создаем папку для сохранения записей, если её нет
        self.output_dir = "recordings"
        os.makedirs(self.output_dir, exist_ok=True)

        # Список для хранения чанков
        self.chunks = []
        self.session_text = ""
        self.session_count = 1

        # Таймер для обновления текста в реальном времени
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_text_display)

        # Аутентификация SaluteSpeech (API)
        self.auth = ""
        self.salute_token = self.get_token(self.auth)

        # Регистрируем шрифт, поддерживающий кириллицу.
        # Используем resource_path, чтобы корректно найти файл при запуске из exe.
        font_path = resource_path("DejaVuSans.ttf")
        pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))

    def populate_device_list(self):
        # Получаем все устройства и добавляем только те, у которых есть входные каналы
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                self.device_selector.addItem(dev['name'], i)

    def get_token(self, auth_token, scope='SALUTE_SPEECH_PERS'):
        rq_uid = str(uuid.uuid4())
        url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'RqUID': rq_uid,
            'Authorization': f'Basic {auth_token}'
        }

        payload = {'scope': scope}

        try:
            response = requests.post(url, headers=headers, data=payload, verify=False)
            response.raise_for_status()
            return response.json().get('access_token')
        except requests.RequestException as e:
            self.text_display.append(f"Ошибка при получении токена: {str(e)}")
            return None

    def start_recording(self):
        self.text_display.append("Запись начата...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.recording = True

        self.chunks = []
        self.session_text = ""
        self.record_thread = Thread(target=self.record)
        self.record_thread.start()

        self.timer.start(1000)

    def record(self):
        while self.recording:
            try:
                ch_duration = float(self.duration_input.text())
            except ValueError:
                ch_duration = 20

            # Получаем индекс выбранного устройства
            selected_device = self.device_selector.currentData()

            # Запись аудио с использованием выбранного устройства
            audio_data = sd.rec(int(ch_duration * self.fs), samplerate=self.fs, channels=self.channels, dtype='int16', device=selected_device)
            sd.wait()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.output_dir, f"chunk_{timestamp}.wav")
            sf.write(filename, audio_data, self.fs)

            self.chunks.append(filename)

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.text_display.append("Запись остановлена.")
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.timer.stop()
            if hasattr(self, "record_thread") and self.record_thread.is_alive():
                self.record_thread.join()
            while self.chunks:
                latest_chunk = self.chunks.pop(0)
                result = self.stt(latest_chunk, self.salute_token)
                if result:
                    self.text_display.append(result)
                    self.session_text += result + "\n"
            self.save_to_pdf()
            self.cleanup_audio_files()

    def update_text_display(self):
        if self.chunks and self.salute_token:
            latest_chunk = self.chunks.pop(0)
            result = self.stt(latest_chunk, self.salute_token)
            if result:
                self.text_display.append(result)
                self.session_text += result + "\n"

    def stt(self, file_path, token):
        url = "https://smartspeech.sber.ru/rest/v1/speech:recognize"

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "audio/x-pcm;bit=16;rate=16000"
        }

        try:
            with open(file_path, "rb") as audio_file:
                audio_data = audio_file.read()

            response = requests.post(url, headers=headers, data=audio_data, verify=False)

            if response.status_code == 200:
                result = response.json()
                return result.get("result", [None])[0]
            else:
                self.text_display.append(f"Ошибка: {response.status_code} {response.text}")
        except Exception as e:
            self.text_display.append(f"Ошибка при распознавании: {str(e)}")

        return None

    def save_to_pdf(self):
        existing_files = [f for f in os.listdir() if f.startswith("Выступление_") and f.endswith(".pdf")]
        if not existing_files:
            index = 1
        else:
            indices = []
            for filename in existing_files:
                try:
                    number = int(filename.split('_')[1].split('.')[0])
                    indices.append(number)
                except ValueError:
                    continue
            index = max(indices) + 1 if indices else 1

        pdf_filename = f"Выступление_{index}.pdf"
        c = canvas.Canvas(pdf_filename, pagesize=A4)
        width, height = A4
        margin = 40
        line_height = 15
        c.setFont("DejaVuSans", 12)

        available_width = width - 2 * margin
        y = height - margin

        for orig_line in self.session_text.splitlines():
            if not orig_line.strip():
                y -= line_height
                continue
            words = orig_line.split()
            current_line = words[0]
            for word in words[1:]:
                test_line = current_line + " " + word
                if pdfmetrics.stringWidth(test_line, "DejaVuSans", 12) <= available_width:
                    current_line = test_line
                else:
                    if y < margin + line_height:
                        c.showPage()
                        c.setFont("DejaVuSans", 12)
                        y = height - margin
                    c.drawString(margin, y, current_line)
                    y -= line_height
                    current_line = word
            if current_line:
                if y < margin + line_height:
                    c.showPage()
                    c.setFont("DejaVuSans", 12)
                    y = height - margin
                c.drawString(margin, y, current_line)
                y -= line_height

        c.save()
        self.text_display.append(f"Сохранено в PDF: {pdf_filename}")

    def cleanup_audio_files(self):
        for file_path in os.listdir(self.output_dir):
            full_path = os.path.join(self.output_dir, file_path)
            if os.path.isfile(full_path) and file_path.endswith('.wav'):
                os.remove(full_path)
        self.text_display.append("Все аудиофайлы удалены.")

    def copy_to_clipboard(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.session_text)
        self.text_display.append("Текст скопирован в буфер обмена.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    sys.exit(app.exec())
