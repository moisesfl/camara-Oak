#Test modelo utilizando YOLO: realmente poderíase facer cunha cámara calquera.Non estou utilizando o modelo convertido .blob

import cv2
from ultralytics import YOLO

# Load the YOLOv5 model (pretrained weights)
model = YOLO('C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/my_model.pt')  # 'yolov5s.pt' is the small version of YOLOv5
labels = model.names

# Load the image
image_path = 'C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/test2.jpeg'
frame = cv2.imread(image_path)

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]


# Perform inference with YOLO
results = model(frame, verbose=False)

detections = results[0].boxes

# Initialize variable for basic object counting example
object_count = 0

# Go through each detection and get bbox coords, confidence, and class
for i in range(len(detections)):

    # Get bounding box coordinates
    # Ultralytics returns results in Tensor format, which have to be converted to a regular Python array
    xyxy_tensor = detections[i].xyxy.cpu() # Detections in Tensor format in CPU memory
    xyxy = xyxy_tensor.numpy().squeeze() # Convert tensors to Numpy array
    xmin, ymin, xmax, ymax = xyxy.astype(int) # Extract individual coordinates and convert to int

    # Get bounding box class ID and name
    classidx = int(detections[i].cls.item())
    classname = labels[classidx]

    # Get bounding box confidence
    conf = detections[i].conf.item()

    # Draw box if confidence threshold is high enough
    if conf > 0.5:

        color = bbox_colors[classidx % 10]
        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)

        label = f'{classname}: {int(conf*100)}%'
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # Get font size
        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED) # Draw white box to put label text in
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) # Draw label text

        # Basic example: count the number of objects in the image
        object_count = object_count + 1

    cv2.imshow('YOLO detection results',frame) # Display image
    if cv2.waitKey(1) == ord('q'):
        break











# import cv2
# import numpy as np

# # Ruta del modelo ONNX
# model_path = 'C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/modelo.onnx'

# # Cargar el modelo ONNX con OpenCV
# net = cv2.dnn.readNetFromONNX(model_path)

# # Leer una imagen de entrada
# image = cv2.imread('test.jpeg')

# # Preprocesar la imagen para que sea compatible con el modelo
# # Convertir la imagen a un blob (formato que OpenCV DNN usa para la entrada)
# blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# # Pasar el blob como entrada a la red
# net.setInput(blob)

# # Obtener la salida del modelo
# output = net.forward()

# # Mostrar el resultado de la inferencia
# print("Output:", output)

# # Mostrar la imagen de entrada
# cv2.imshow("Input Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()











# import cv2
# import numpy as np

# # Load OpenVINO model into OpenCV
# net = cv2.dnn.readNet("model.xml", "model.bin")

# # Load an image
# image = cv2.imread("test.jpg")
# blob = cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True, crop=False)

# # Run inference
# net.setInput(blob)
# output = net.forward()

# print("Model Output:", output)