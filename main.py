# Instalar las librerías necesarias (ejecuta en tu entorno)
# pip install roboflow ultralytics opencv-python pyyaml

import os
import random
import shutil
import yaml
from pathlib import Path
from roboflow import Roboflow

rf = Roboflow(api_key="2DfaLwuoO0656xAnYn8j")
project = rf.workspace("littlefruitproject").project("little-fruit")
version = project.version(3)
dataset = version.download("yolov8") 

# La ruta base del dataset descargando (little-fruit-3)
dataset_path = Path(dataset.location)

# Los datos originales están en la carpeta 'train'
orig_images_dir = dataset_path / "train" / "images"
orig_labels_dir = dataset_path / "train" / "labels"

# Crear nuevas carpetas para la división dentro de 'images' y 'labels'
new_train_images = dataset_path / "images" / "train_split"
new_val_images   = dataset_path / "images" / "val"
new_test_images  = dataset_path / "images" / "test"

new_train_labels = dataset_path / "labels" / "train_split"
new_val_labels   = dataset_path / "labels" / "val"
new_test_labels  = dataset_path / "labels" / "test"

for folder in [new_train_images, new_val_images, new_test_images, 
               new_train_labels, new_val_labels, new_test_labels]:
    folder.mkdir(parents=True, exist_ok=True)


# dividir el dataset (80% train, 10% val, 10% test)

all_images = list(orig_images_dir.glob("*.*"))
n_images = len(all_images)
print(f"Total de imágenes encontradas: {n_images}")

# Mezclar aleatoriamente las imágenes
random.shuffle(all_images)

train_count = int(n_images * 0.8)
val_count   = int(n_images * 0.1)
# El resto para test
test_count  = n_images - train_count - val_count

train_imgs = all_images[:train_count]
val_imgs   = all_images[train_count:train_count + val_count]
test_imgs  = all_images[train_count + val_count:]

def move_files(file_list, dest_img_dir, dest_label_dir):
    for img_path in file_list:
        # Mover la imagen a la carpeta destino
        shutil.move(str(img_path), str(dest_img_dir / img_path.name))
        # Mover la etiqueta correspondiente (mismo nombre con extensión .txt)
        label_path = orig_labels_dir / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.move(str(label_path), str(dest_label_dir / label_path.name))

move_files(train_imgs, new_train_images, new_train_labels)
move_files(val_imgs, new_val_images, new_val_labels)
move_files(test_imgs, new_test_images, new_test_labels)

# Cargar el archivo data.yaml original y actualizar las rutas para cada split
data_yaml_path = dataset_path / "data.yaml"
with open(data_yaml_path, "r") as f:
    data_config = yaml.safe_load(f)

data_config["train"] = str(new_train_images)
data_config["val"]   = str(new_val_images)
data_config["test"]  = str(new_test_images)
data_config["nc"]    = 1
data_config["names"] = ["fruto"]

with open(data_yaml_path, "w") as f:
    yaml.dump(data_config, f)

print("Archivo data.yaml actualizado.")

# =============================================================================
# 5. Entrenar el modelo YOLOv8 y guardar el mejor modelo
# =============================================================================
from ultralytics import YOLO

# modelo preentrenado
model = YOLO("yolov8n.pt")

# Entrenar el modelo ahora si
train_results = model.train(data=str(data_yaml_path), epochs=40, imgsz=640, device="mps")

# Guardar el mejor modelo (por defecto se guarda en runs/detect/exp/weights/best.pt)
best_weights_path = Path("runs/detect/exp/weights/best.pt")
if best_weights_path.exists():
    shutil.copy(best_weights_path, "modelo_final.pt")
    print("Modelo guardado como 'modelo_final.pt'.")
else:
    print("No se encontró el archivo 'best.pt'. Verifica el directorio de entrenamiento.")

val_results = model.val(data=str(data_yaml_path))
# print(val_results)
