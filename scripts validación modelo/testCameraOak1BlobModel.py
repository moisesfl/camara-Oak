#Cargar blobl model y ver en tiempo real la detección de objetos
configPath = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_puerta/result (2)/my_model.json"
#!/usr/bin/env python3
#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import json
import os

class ModelConfig:
    def __init__(self, path):
        with open(path, 'r') as f:
            self.data = json.load(f)

    @property
    def model_xml(self):
        return self.data['model']['xml']

    @property
    def model_bin(self):
        return self.data['model']['bin']

    @property
    def input_size(self):
        return self.data['nn_config']['input_size']

    @property
    def output_format(self):
        return self.data['nn_config']['output_format']

    @property
    def nn_family(self):
        return self.data['nn_config']['NN_family']

    @property
    def num_classes(self):
        return self.data['nn_config']['NN_specific_metadata']['classes']

    @property
    def confidence_threshold(self):
        return self.data['nn_config']['NN_specific_metadata']['confidence_threshold']

    @property
    def iou_threshold(self):
        return self.data['nn_config']['NN_specific_metadata']['iou_threshold']

    @property
    def labels(self):
        return self.data['mappings']['labels']

    def __repr__(self):
        return f"<ModelConfig: {self.model_xml}, classes={self.num_classes}, labels={self.labels}>"



import depthai as dai
import cv2
import numpy as np
import json
import time

# === CONFIGURAR RUTAS ===
BLOB_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_puerta/result (2)/my_model_openvino_2022.1_6shave.blob"
IMAGE_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_puerta/dataset/imagenes_redimensionadas_dataset_1/dataset_01_G30_E35m_img_0064.jpg"
JSON_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_puerta/result (2)/my_model.json"


image_folder = "ruta/a/carpeta_de_imagenes"

# === FUNCIONES ===
def to_planar(arr):
    return arr.transpose(2, 0, 1).flatten().tolist()

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# === CARGAR CONFIGURACIÓN DEL MODELO (.json) ===
with open(JSON_PATH, 'r') as f:
    config = json.load(f)

input_size = tuple(map(int, config['nn_config']['input_size'].split('x')))
labels = config['mappings']['labels']
conf_threshold = float(config['nn_config']['NN_specific_metadata']['confidence_threshold'])
iou_threshold = float(config['nn_config']['NN_specific_metadata']['iou_threshold'])
num_classes = int(config['nn_config']['NN_specific_metadata']['classes'])


# === CREAR PIPELINE ===
pipeline = dai.Pipeline()

xin = pipeline.create(dai.node.XLinkIn)
xin.setStreamName("inFrame")

nn = pipeline.create(dai.node.YoloDetectionNetwork)
nn.setBlobPath(BLOB_PATH)
nn.setConfidenceThreshold(conf_threshold)
nn.setNumClasses(num_classes)
nn.setCoordinateSize(4)
nn.setIouThreshold(iou_threshold)
nn.input.setBlocking(True)
nn.input.setQueueSize(1)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("nn")

# ENLACE
xin.out.link(nn.input)
nn.out.link(xout.input)

# === ENVIAR IMAGEN Y RECIBIR RESULTADOS ===
with dai.Device(pipeline) as device:
    input_queue = device.getInputQueue("inFrame")
    output_queue = device.getOutputQueue("nn")
    


    # # === CARGAR Y PREPARAR IMAGEN ===
    # image = cv2.imread(IMAGE_PATH)
    # if image is None:
    #     raise Exception(f"No se pudo cargar la imagen: {IMAGE_PATH}")

    # image_resized = cv2.resize(image, input_size)
    # image_planar = to_planar(image_resized)


    # # Crear objeto ImgFrame
    # img_frame = dai.ImgFrame()
    # img_frame.setData(image_planar)
    # img_frame.setWidth(input_size[0])
    # img_frame.setHeight(input_size[1])
    # img_frame.setType(dai.ImgFrame.Type.BGR888p)

    # input_queue.send(img_frame)
    # result = output_queue.get()

    # # Dibujar resultados
    # for det in result.detections:
    #     bbox = frame_norm(image, (det.xmin, det.ymin, det.xmax, det.ymax))
    #     label = labels[det.label] if det.label < len(labels) else str(det.label)
    #     conf = int(det.confidence * 100)

    #     cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    #     cv2.putText(image, f"{label} {conf}%", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # # Mostrar
    # cv2.imshow("Detecciones", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    image_folder = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_puerta/dataset/imagenes_redimensionadas_dataset_1/val/negras"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    for filename in image_files:
        path = os.path.join(image_folder, filename)
        image = cv2.imread(path)

           # === CARGAR Y PREPARAR IMAGEN ===

        if image is None:
            raise Exception(f"No se pudo cargar la imagen: {IMAGE_PATH}")

        image_resized = cv2.resize(image, input_size)
        image_planar = to_planar(image_resized)


        # Crear objeto ImgFrame
        img_frame = dai.ImgFrame()
        img_frame.setData(image_planar)
        img_frame.setWidth(input_size[0])
        img_frame.setHeight(input_size[1])
        img_frame.setType(dai.ImgFrame.Type.BGR888p)

        input_queue.send(img_frame)
        result = output_queue.get()

        # Dibujar resultados
        for det in result.detections:
            bbox = frame_norm(image, (det.xmin, det.ymin, det.xmax, det.ymax))
            label = labels[det.label] if det.label < len(labels) else str(det.label)
            conf = int(det.confidence * 100)

            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf}%", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Mostrar
        cv2.imshow("Detecciones", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
