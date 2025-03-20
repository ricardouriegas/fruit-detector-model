import sys
import cv2
import torch
import numpy as np
import os
import time
from collections import deque
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QSizePolicy, QSlider, QComboBox
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class FruitTracker:
    # Inicializa el rastreador de frutos.
    def __init__(self, distance_threshold=50, confidence_threshold=0.58, confidence_history_size=5):
        self.fruit_count = 0
        self.tracked_fruits = []
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.next_id = 0
        self.confidence_history_size = confidence_history_size
        self.fruit_confidence_history = {}
        self.last_seen_threshold = 30
        
    # Verifica si el fruto detectado es nuevo.
    def is_new_fruit(self, x, y):
        center_x = (x + x) // 2
        center_y = (y + y) // 2
        closest_fruit = None
        min_distance = float('inf')
        current_fruits = []
        for fruit in self.tracked_fruits:
            fx, fy, fruit_id, conf, last_seen = fruit
            if last_seen <= self.last_seen_threshold:
                current_fruits.append(fruit)
                distance = np.sqrt((center_x - fx)**2 + (center_y - fy)**2)
                if distance < self.distance_threshold and distance < min_distance:
                    min_distance = distance
                    closest_fruit = fruit
        self.tracked_fruits = current_fruits
        if closest_fruit:
            return False, closest_fruit[2]
        fruit_id = self.next_id
        self.next_id += 1
        return True, fruit_id
    
    # Agrega un fruto para seguimiento.
    def add_fruit(self, x, y, confidence):
        center_x = (x + x) // 2
        center_y = (y + y) // 2
        is_new, fruit_id = self.is_new_fruit(center_x, center_y)
        if fruit_id not in self.fruit_confidence_history:
            self.fruit_confidence_history[fruit_id] = deque(maxlen=self.confidence_history_size)
        self.fruit_confidence_history[fruit_id].append(confidence)
        avg_confidence = sum(self.fruit_confidence_history[fruit_id]) / len(self.fruit_confidence_history[fruit_id])
        if avg_confidence < self.confidence_threshold:
            return False, None, avg_confidence
        updated = False
        for i, (fx, fy, fid, _, last_seen) in enumerate(self.tracked_fruits):
            if fid == fruit_id:
                self.tracked_fruits[i] = (center_x, center_y, fid, avg_confidence, 0)
                updated = True
                break
        if not updated:
            if is_new:
                self.tracked_fruits.append((center_x, center_y, fruit_id, avg_confidence, 0))
                self.fruit_count += 1
        return is_new, fruit_id, avg_confidence
    
    # Incrementa el contador "last_seen" de cada fruto.
    def update_tracking(self):
        for i in range(len(self.tracked_fruits)):
            x, y, id, conf, last_seen = self.tracked_fruits[i]
            self.tracked_fruits[i] = (x, y, id, conf, last_seen + 1)
    
    # Reinicia contador y seguimiento.
    def reset(self):
        self.fruit_count = 0
        self.tracked_fruits = []
        self.next_id = 0
        self.fruit_confidence_history = {}
        
    # Reinicia el seguimiento para una nueva rotación.
    def reset_for_rotation(self):
        self.tracked_fruits = []
        self.fruit_confidence_history = {}
        
    # Actualiza el umbral de confianza.
    def set_confidence_threshold(self, value):
        self.confidence_threshold = value

class MainWindow(QMainWindow):
    # Inicializa la ventana principal y la interfaz.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contador de Fresas en Rotación")
        self.setMinimumSize(800, 600)
        self.resize(1024, 768)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.counter_label = QLabel("Fresas detectadas: 0")
        self.counter_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.threshold_label = QLabel(f"Umbral de confianza: 0.58")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(90)
        self.threshold_slider.setValue(58)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        self.speed_label = QLabel("Velocidad de procesamiento:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Rápido (30 FPS)", "Normal (15 FPS)", "Lento (5 FPS)", "Muy lento (2 FPS)"])
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.currentIndexChanged.connect(self.update_processing_speed)
        self.start_button = QPushButton("Iniciar Detección")
        self.start_button.clicked.connect(self.toggle_detection)
        self.reset_button = QPushButton("Reiniciar Contador")
        self.reset_button.clicked.connect(self.reset_counter)
        self.rotation_button = QPushButton("Nueva Rotación")
        self.rotation_button.clicked.connect(self.handle_rotation)


        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.rotation_button)

        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_combo)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.counter_label)
        main_layout.addLayout(threshold_layout)
        main_layout.addLayout(speed_layout)
        main_layout.addLayout(button_layout)


        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # esto es para que funcione en mac o con cuda (gpu)
        is_mac_silicon = torch.backends.mps.is_available()
        self.device = 'mps' if is_mac_silicon else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")


        self.confidence_threshold = 0.58
        self.fruit_tracker = FruitTracker(distance_threshold=50, confidence_threshold=self.confidence_threshold, confidence_history_size=5)

        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detecting = False
        self.frame_width = 640
        self.frame_height = 480
        self.yolo_threshold = 0.58
        self.process_every_n_frames = 2
        self.frame_count = 0
        self.fps_values = {0: 30, 1: 15, 2: 5, 3: 2}
        self.last_process_time = time.time()
        self.motion_history = []
        self.rotation_detected = False
        self.motion_threshold = 600
        self.rotation_status_label = QLabel("Estado de rotación: Esperando movimiento")
        main_layout.addWidget(self.rotation_status_label)

    # Alterna la detección en vivo.
    def toggle_detection(self):
        if self.detecting:
            self.timer.stop()
            self.start_button.setText("Iniciar Detección")
            self.detecting = False
        else:
            self.update_processing_speed()
            self.timer.start(1000 // self.fps_values[self.speed_combo.currentIndex()])
            self.start_button.setText("Detener Detección")
            self.detecting = True

    # Reinicia el contador de frutos.
    def reset_counter(self):
        self.fruit_tracker.reset()
        self.counter_label.setText(f"Fresas detectadas: 0")

    # Maneja la rotación completada y reinicia el seguimiento.
    def handle_rotation(self):
        self.fruit_tracker.reset_for_rotation()
        print("Nueva rotación iniciada - tracking reiniciado")

    # Actualiza el umbral de confianza según el slider.
    def update_threshold(self):
        threshold_value = self.threshold_slider.value() / 100.0
        self.confidence_threshold = threshold_value
        self.threshold_label.setText(f"Umbral de confianza: {threshold_value:.2f}")
        self.fruit_tracker.set_confidence_threshold(threshold_value)
        
    # Actualiza la velocidad de procesamiento según la selección.
    def update_processing_speed(self):
        index = self.speed_combo.currentIndex()
        fps = self.fps_values[index]
        if index == 0:
            self.process_every_n_frames = 1
        elif index == 1:
            self.process_every_n_frames = 2
        elif index == 2:
            self.process_every_n_frames = 6
        else:
            self.process_every_n_frames = 15
        if self.detecting:
            self.timer.stop()
            self.timer.start(1000 // fps)

    # Verifica si se completó una rotación.
    def check_rotation(self, frame):
        if self.frame_count % 5 != 0:
            return False
        if len(self.motion_history) == 0:
            self.motion_history.append(frame)
            return False
        
        first_frame = self.motion_history[0]

        frame_delta = cv2.absdiff(first_frame, frame)
        frame_delta_gray = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)

        thresh = cv2.threshold(frame_delta_gray, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_value = 0

        for contour in contours:
            motion_value += cv2.contourArea(contour)
        if len(self.motion_history) >= 20:
            self.motion_history = self.motion_history[1:] + [frame]
        else:
            self.motion_history.append(frame)
        if motion_value < self.motion_threshold and self.rotation_detected:
            self.rotation_detected = False
            print("Rotación completada detectada")
            self.rotation_status_label.setText("Estado de rotación: Rotación completada, reiniciando tracking")
            self.handle_rotation()
            self.motion_history = [frame]
            return True
        if motion_value > self.motion_threshold and not self.rotation_detected:
            self.rotation_detected = True
            self.rotation_status_label.setText("Estado de rotación: Rotación en progreso")
        cv2.putText(frame, f"Movimiento: {motion_value}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return False

    # Captura, procesa el frame y actualiza la interfaz.
    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        try:
            self.frame_height, self.frame_width = frame.shape[:2]
            self.frame_count += 1
            self.fruit_tracker.update_tracking()
            self.check_rotation(frame)
            
            process_this_frame = (self.frame_count % self.process_every_n_frames == 0)

            if process_this_frame:
                current_time = time.time()
                process_interval = current_time - self.last_process_time
                self.last_process_time = current_time
                results = self.model(frame, conf=self.yolo_threshold)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        is_new, fruit_id, avg_conf = self.fruit_tracker.add_fruit(x1, y1, conf)
                        if avg_conf < self.confidence_threshold:
                            color = (128, 128, 128)
                        else:
                            color = (0, 255, 0) if is_new else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label_text = f"Conf: {avg_conf:.2f}"
                        if avg_conf >= self.confidence_threshold and fruit_id is not None:
                            label_text = f"Fresa #{fruit_id}: {avg_conf:.2f}"
                        cv2.putText(frame, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            self.counter_label.setText(f"Fresas detectadas: {self.fruit_tracker.fruit_count}")
            
            fps_text = f"FPS: {self.fps_values[self.speed_combo.currentIndex()]}, Procesando: 1/{self.process_every_n_frames} frames"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            label_size = self.video_label.size()
            frame_aspect = self.frame_width / self.frame_height
            label_aspect = label_size.width() / label_size.height()

            if frame_aspect > label_aspect:
                display_width = label_size.width()
                display_height = int(display_width / frame_aspect)
            else:
                display_height = label_size.height()
                display_width = int(display_height * frame_aspect)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if display_width > 0 and display_height > 0:
                frame_rgb = cv2.resize(frame_rgb, (display_width, display_height))

            h, w, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception as e:
            print(f"Error en detección: {e}")

    # Maneja el evento de redimensionamiento.
    def resizeEvent(self, event):
        super().resizeEvent(event)

    # Libera la captura de video al cerrar la ventana.
    def closeEvent(self, event):
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
