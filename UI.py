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
    def __init__(self, distance_threshold=50, confidence_threshold=0.58, confidence_history_size=5):
        self.fruit_count = 0
        self.tracked_fruits = []  # Lista de frutos ya contados [(x, y, id, confidence, last_seen), ...]
        self.distance_threshold = distance_threshold
        self.confidence_threshold = confidence_threshold
        self.next_id = 0
        self.confidence_history_size = confidence_history_size
        self.fruit_confidence_history = {}  # {id: deque of recent confidence values}
        self.last_seen_threshold = 30  # Frames after which to consider a fruit inactive
        
    def is_new_fruit(self, x, y):
        """Determina si un fruto detectado es nuevo o ya ha sido contado"""
        center_x = (x + x) // 2
        center_y = (y + y) // 2
        
        closest_fruit = None
        min_distance = float('inf')
        
        # Eliminar frutos que no se han visto recientemente
        current_fruits = []
        for fruit in self.tracked_fruits:
            fx, fy, fruit_id, conf, last_seen = fruit
            if last_seen <= self.last_seen_threshold:
                current_fruits.append(fruit)
                
                # Calcular distancia euclidiana
                distance = np.sqrt((center_x - fx)**2 + (center_y - fy)**2)
                if distance < self.distance_threshold and distance < min_distance:
                    min_distance = distance
                    closest_fruit = fruit
                    
        self.tracked_fruits = current_fruits
        
        if closest_fruit:
            return False, closest_fruit[2]  # Fruto ya contado, retornar su ID
        
        # Es un nuevo fruto
        fruit_id = self.next_id
        self.next_id += 1
        return True, fruit_id
    
    def add_fruit(self, x, y, confidence):
        """Añade un nuevo fruto al seguimiento y aumenta el contador"""
        # Actualizar centro
        center_x = (x + x) // 2
        center_y = (y + y) // 2
        
        # Comprobar si es un fruto ya existente o nuevo
        is_new, fruit_id = self.is_new_fruit(center_x, center_y)
        
        # Actualizar historial de confianza
        if fruit_id not in self.fruit_confidence_history:
            self.fruit_confidence_history[fruit_id] = deque(maxlen=self.confidence_history_size)
        
        self.fruit_confidence_history[fruit_id].append(confidence)
        
        # Calcular confianza promedio
        avg_confidence = sum(self.fruit_confidence_history[fruit_id]) / len(self.fruit_confidence_history[fruit_id])
        
        # Solo procesar detecciones con confianza promedio por encima del umbral
        if avg_confidence < self.confidence_threshold:
            return False, None, avg_confidence
        
        # Actualizar o agregar el fruto
        updated = False
        for i, (fx, fy, fid, _, last_seen) in enumerate(self.tracked_fruits):
            if fid == fruit_id:
                # Actualizar posición y reiniciar contador de "último visto"
                self.tracked_fruits[i] = (center_x, center_y, fid, avg_confidence, 0)
                updated = True
                break
                
        if not updated:
            # Si es un nuevo fruto con confianza suficiente
            if is_new:
                self.tracked_fruits.append((center_x, center_y, fruit_id, avg_confidence, 0))
                self.fruit_count += 1
                
        return is_new, fruit_id, avg_confidence
    
    def update_tracking(self):
        """Incrementa los contadores de 'último visto' para todos los frutos"""
        for i in range(len(self.tracked_fruits)):
            x, y, id, conf, last_seen = self.tracked_fruits[i]
            self.tracked_fruits[i] = (x, y, id, conf, last_seen + 1)
    
    def reset(self):
        """Reinicia el contador y el seguimiento"""
        self.fruit_count = 0
        self.tracked_fruits = []
        self.next_id = 0
        self.fruit_confidence_history = {}
        
    def reset_for_rotation(self):
        """Mantiene el conteo pero reinicia el seguimiento para nueva rotación"""
        # Mantenemos el count pero limpiamos el tracking
        self.tracked_fruits = []
        self.fruit_confidence_history = {}
        
    def set_confidence_threshold(self, value):
        """Actualiza el umbral de confianza"""
        self.confidence_threshold = value

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contador de Fresas en Rotación")
        
        # Configurar tamaño inicial y política de redimensionamiento
        self.setMinimumSize(800, 600)  # Tamaño mínimo
        self.resize(1024, 768)  # Tamaño inicial razonable
        
        # Widget para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)  # Tamaño mínimo para el video
        self.video_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Etiqueta para mostrar el contador de frutos
        self.counter_label = QLabel("Fresas detectadas: 0")
        self.counter_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.counter_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Control de umbral de confianza
        self.threshold_label = QLabel(f"Umbral de confianza: 0.58")
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setMinimum(10)  # 0.1
        self.threshold_slider.setMaximum(90)  # 0.9
        self.threshold_slider.setValue(58)    # 0.58 por defecto
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        # Selector de velocidad de procesamiento
        self.speed_label = QLabel("Velocidad de procesamiento:")
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["Rápido (30 FPS)", "Normal (15 FPS)", "Lento (5 FPS)", "Muy lento (2 FPS)"])
        self.speed_combo.setCurrentIndex(1)  # Normal por defecto
        self.speed_combo.currentIndexChanged.connect(self.update_processing_speed)
        
        # Botones para controlar la detección
        self.start_button = QPushButton("Iniciar Detección")
        self.start_button.clicked.connect(self.toggle_detection)
        
        self.reset_button = QPushButton("Reiniciar Contador")
        self.reset_button.clicked.connect(self.reset_counter)
        
        self.rotation_button = QPushButton("Nueva Rotación")
        self.rotation_button.clicked.connect(self.handle_rotation)
        
        # Layout para botones
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addWidget(self.rotation_button)
        
        # Layout para umbral
        threshold_layout = QHBoxLayout()
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_slider)
        
        # Layout para velocidad
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.speed_label)
        speed_layout.addWidget(self.speed_combo)
        
        # Layout principal
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addWidget(self.counter_label)
        main_layout.addLayout(threshold_layout)
        main_layout.addLayout(speed_layout)
        main_layout.addLayout(button_layout)
        
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Configuración del dispositivo: usar Apple Silicon, CUDA o CPU según disponibilidad
        is_mac_silicon = torch.backends.mps.is_available()
        self.device = 'mps' if is_mac_silicon else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Cargar el modelo PyTorch
        try:
            self.model = YOLO("best.pt")
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
        
        # Inicializar el rastreador de frutas con umbral de confianza
        self.confidence_threshold = 0.58  # valor predeterminado
        self.fruit_tracker = FruitTracker(
            distance_threshold=50, 
            confidence_threshold=self.confidence_threshold,
            confidence_history_size=5
        )
        
        # Abrir captura de video (la cámara por defecto, índice 0)
        self.cap = cv2.VideoCapture(0)
        
        # Timer para actualizar los frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detecting = False

        # Preparar variables para el redimensionamiento
        self.frame_width = 640
        self.frame_height = 480

        # Umbral para YOLO
        self.yolo_threshold = 0.58  # Confianza base para YOLO
        
        # Control de velocidad de procesamiento
        self.process_every_n_frames = 2  # Procesar cada N frames
        self.frame_count = 0
        self.fps_values = {0: 30, 1: 15, 2: 5, 3: 2}  # Índice del combo -> FPS
        
        # Detección de rotación
        self.last_process_time = time.time()
        self.motion_history = []
        self.rotation_detected = False
        self.motion_threshold = 600  # Umbral para detectar movimiento significativo
        self.rotation_status_label = QLabel("Estado de rotación: Esperando movimiento")
        main_layout.addWidget(self.rotation_status_label)

    def toggle_detection(self):
        """Inicia o detiene la detección en vivo."""
        if self.detecting:
            self.timer.stop()
            self.start_button.setText("Iniciar Detección")
            self.detecting = False
        else:
            self.update_processing_speed()  # Configurar FPS según selección
            self.timer.start(1000 // self.fps_values[self.speed_combo.currentIndex()])
            self.start_button.setText("Detener Detección")
            self.detecting = True

    def reset_counter(self):
        """Reinicia el contador de frutos"""
        self.fruit_tracker.reset()
        self.counter_label.setText(f"Fresas detectadas: 0")

    def handle_rotation(self):
        """Manejar cuando la planta completa una rotación"""
        self.fruit_tracker.reset_for_rotation()
        print("Nueva rotación iniciada - tracking reiniciado")

    def update_threshold(self):
        """Actualiza el umbral de confianza basado en el valor del slider"""
        threshold_value = self.threshold_slider.value() / 100.0
        self.confidence_threshold = threshold_value
        self.threshold_label.setText(f"Umbral de confianza: {threshold_value:.2f}")
        self.fruit_tracker.set_confidence_threshold(threshold_value)
        
    def update_processing_speed(self):
        """Actualiza la velocidad de procesamiento basado en la selección"""
        index = self.speed_combo.currentIndex()
        fps = self.fps_values[index]
        
        # Ajustar el proceso_every_n_frames basado en la velocidad seleccionada
        if index == 0:  # Rápido
            self.process_every_n_frames = 1
        elif index == 1:  # Normal
            self.process_every_n_frames = 2
        elif index == 2:  # Lento
            self.process_every_n_frames = 6
        else:  # Muy lento
            self.process_every_n_frames = 15
            
        # Si la detección ya está activa, actualizar el timer
        if self.detecting:
            self.timer.stop()
            self.timer.start(1000 // fps)

    def check_rotation(self, frame):
        """Detecta si la planta ha completado una rotación basado en el movimiento"""
        # Calcular el flujo óptico o cambio de imagen sólo cada cierto número de frames
        if self.frame_count % 5 != 0:
            return False
            
        # Si es el primer frame, inicializar
        if len(self.motion_history) == 0:
            self.motion_history.append(frame)
            return False
            
        # Calcular diferencia con el primer frame de referencia
        first_frame = self.motion_history[0]
        frame_delta = cv2.absdiff(first_frame, frame)
        # Convertir diferencia a escala de grises para umbralización
        frame_delta_gray = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(frame_delta_gray, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Dilatación para mejorar la detección
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Encontrar contornos para analizar el movimiento
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_value = 0
        for contour in contours:
            motion_value += cv2.contourArea(contour)
        
        # Mantener un historial del movimiento
        if len(self.motion_history) >= 20:  # Mantener un historial limitado
            self.motion_history = self.motion_history[1:] + [frame]
        else:
            self.motion_history.append(frame)
            
        # Comprobar si el movimiento ha descendido después de un pico
        # Esto puede indicar que la rotación se ha completado
        if motion_value < self.motion_threshold and self.rotation_detected:
            self.rotation_detected = False
            print("Rotación completada detectada")
            self.rotation_status_label.setText("Estado de rotación: Rotación completada, reiniciando tracking")
            self.handle_rotation()
            # Actualizar el marco de referencia
            self.motion_history = [frame]
            return True
            
        # Detectar el inicio de la rotación
        if motion_value > self.motion_threshold and not self.rotation_detected:
            self.rotation_detected = True
            self.rotation_status_label.setText("Estado de rotación: Rotación en progreso")
        
        # Mostrar valor de movimiento en la interfaz para depuración
        cv2.putText(frame, f"Movimiento: {motion_value}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        return False

    def update_frame(self):
        """Captura un frame, realiza la detección y actualiza la interfaz."""
        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            # Guardar las dimensiones originales del frame
            self.frame_height, self.frame_width = frame.shape[:2]
            
            # Incrementar contador de frames
            self.frame_count += 1
            
            # Actualizar el tracking para todos los frutos (incrementar last_seen)
            self.fruit_tracker.update_tracking()
            
            # Verificar si ha completado una rotación - siempre activo
            self.check_rotation(frame)
            
            # Procesar sólo cada N frames para reducir carga y falsos positivos
            process_this_frame = (self.frame_count % self.process_every_n_frames == 0)
            
            if process_this_frame:
                # Registrar el tiempo de proceso
                current_time = time.time()
                process_interval = current_time - self.last_process_time
                self.last_process_time = current_time
                
                # Realizar la inferencia con YOLOv8
                results = self.model(frame, conf=self.yolo_threshold)
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Extraer coordenadas y confianza
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        
                        # Comprobar la clase (asegurarse que sea fresa)
                        cls = int(box.cls[0])
                        
                        # Sólo procesar si la clase corresponde a fresas (normalmente clase 0)
                        # Pero podría variar dependiendo de tu modelo
                        
                        # Verificar si es un nuevo fruto con confianza suavizada
                        is_new, fruit_id, avg_conf = self.fruit_tracker.add_fruit(x1, y1, conf)
                        
                        # Color según el estado
                        if avg_conf < self.confidence_threshold:
                            color = (128, 128, 128)  # Gris para detecciones bajo el umbral
                        else:
                            color = (0, 255, 0) if is_new else (0, 165, 255)  # Verde si es nuevo, naranja si ya contado
                        
                        # Dibujar rectángulo y etiqueta
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        label_text = f"Conf: {avg_conf:.2f}"
                        if avg_conf >= self.confidence_threshold:
                            if fruit_id is not None:
                                label_text = f"Fresa #{fruit_id}: {avg_conf:.2f}"
                        
                        cv2.putText(frame, label_text, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Actualizar contador en la interfaz
            self.counter_label.setText(f"Fresas detectadas: {self.fruit_tracker.fruit_count}")
            
            # Mostrar FPS e información
            fps_text = f"FPS: {self.fps_values[self.speed_combo.currentIndex()]}, Procesando: 1/{self.process_every_n_frames} frames"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Redimensionar el frame al tamaño disponible manteniendo la relación de aspecto
            label_size = self.video_label.size()
            
            # Calcular la relación de aspecto
            frame_aspect = self.frame_width / self.frame_height
            label_aspect = label_size.width() / label_size.height()
            
            # Determinar las dimensiones de escalado
            if frame_aspect > label_aspect:
                # Limitado por el ancho
                display_width = label_size.width()
                display_height = int(display_width / frame_aspect)
            else:
                # Limitado por la altura
                display_height = label_size.height()
                display_width = int(display_height * frame_aspect)
            
            # Convertir frame para mostrar en Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Redimensionar el frame
            if display_width > 0 and display_height > 0:  # Prevenir dimensiones inválidas
                frame_rgb = cv2.resize(frame_rgb, (display_width, display_height))
            
            h, w, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            print(f"Error en detección: {e}")

    def resizeEvent(self, event):
        """Manejar el evento de redimensionamiento de la ventana"""
        super().resizeEvent(event)

    def closeEvent(self, event):
        """Al cerrar la ventana, liberar la captura de video."""
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
