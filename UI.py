import sys
import cv2
import torch
import numpy as np
import os
from ultralytics import YOLO
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Detección en Vivo de Frutos")
        
        # Widget para mostrar el video
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Botón para iniciar/detener la detección
        self.start_button = QPushButton("Iniciar Detección")
        self.start_button.clicked.connect(self.toggle_detection)
        
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.start_button)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        # Configuración del dispositivo: usar Apple Silicon, CUDA o CPU según disponibilidad
        is_mac_silicon = torch.backends.mps.is_available()
        self.device = 'mps' if is_mac_silicon else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Cargar el modelo ONNX
        try:
            model_path = "best1.onnx"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
                
            print(f"Cargando modelo ONNX desde: {model_path}")
            self.model = YOLO(model_path)
            print("Modelo ONNX cargado exitosamente")
        except Exception as e:
            print(f"Error al cargar el modelo ONNX: {e}")
            print("Intentando cargar modelo PyTorch de respaldo...")
            try:
                self.model = YOLO("best.pt")
            except Exception as e:
                print(f"Error al cargar modelo de respaldo: {e}")
                self.model = YOLO("yolov8n.pt")
        
        # Abrir captura de video (la cámara por defecto, índice 0)
        self.cap = cv2.VideoCapture(0)
        
        # Timer para actualizar los frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.detecting = False

    def toggle_detection(self):
        """Inicia o detiene la detección en vivo."""
        if self.detecting:
            self.timer.stop()
            self.start_button.setText("Iniciar Detección")
            self.detecting = False
        else:
            self.timer.start(30)  # Aproximadamente 30 fps
            self.start_button.setText("Detener Detección")
            self.detecting = True

    def update_frame(self):
        """Captura un frame, realiza la detección y actualiza la interfaz."""
        ret, frame = self.cap.read()
        if not ret:
            return

        try:
            # Realizar la inferencia con YOLOv8
            results = self.model(frame, conf=0.25)  # Añadir umbral de confianza
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Extraer coordenadas y confianza
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    # Dibujar rectángulo y etiqueta
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Fruta: {conf:.2f}", (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convertir frame para mostrar en Qt
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            qt_image = QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
            
        except Exception as e:
            print(f"Error en detección: {e}")

    def closeEvent(self, event):
        """Al cerrar la ventana, liberar la captura de video."""
        self.cap.release()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
