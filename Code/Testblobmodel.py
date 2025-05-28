import depthai as dai
import numpy as np
import cv2

# Ruta de tu modelo blob
##MODEL_BLOB_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/modelo.blob"
#MODEL_BLOB_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/Oak-1-lite/Code/depthai-main/depthai-main/resources/nn/custom_model - copia/custom_model.blob"

MODEL_BLOB_PATH = "C:/Users/moifra3484/Downloads/result2/my_model_openvino_2022.1_6shave.blob"

# Ruta del modelo ONNX
##model_path = 'model.onnx'
# Cargar el modelo ONNX con OpenCV
##net = cv2.dnn.readNetFromONNX(model_path)

# Crear el pipeline de DepthAI
pipeline = dai.Pipeline()

# Nodo de cámara
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(320, 320)  # Ajusta según el modelo
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

# Nodo de red neuronal
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(MODEL_BLOB_PATH)

# Conectar la cámara a la red neuronal
cam_rgb.preview.link(nn.input)

# Salida de video y red neuronal
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")
cam_rgb.preview.link(xout_video.input)

xout_nn = pipeline.create(dai.node.XLinkOut)
xout_nn.setStreamName("nn")
nn.out.link(xout_nn.input)


detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
detection_nn.setBlobPath(MODEL_BLOB_PATH)
# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(0.5)


# Softmax for final logits
def softmax(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)


# Conectar a la cámara y ejecutar el pipeline
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue("video", maxSize=1, blocking=False)
    nn_queue = device.getOutputQueue("nn", maxSize=1, blocking=False)

    while True:
        # Obtener la imagen
        frame = video_queue.get().getCvFrame()
        #frame = cv2.imread("C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/test.jpeg")

        # Obtener inferencias
        in_nn = nn_queue.tryGet()

        if in_nn:

            # logits = np.array(in_nn.getLayerFp16("output"))  # Expect length = 8 if single 8-class output
            # probabilities   = softmax(logits)
            # predicted_class = np.argmax(probabilities)
            data = np.array(in_nn.getFirstLayerFp16())  # Convertir salida a numpy

            print("NN Raw Output:", data[:20])
            print("Output Shape:", data.shape)

            detections = in_nn.detections
            detections = []
            for i in range(0, len(data), 7):  # Suponiendo que cada detección usa 7 valores
                label = int(data[i])  # Clase detectada
                conf = data[i + 1]  # Confianza
                x_min, y_min, x_max, y_max = data[i + 3 : i + 7]  # Coordenadas normalizadas

                if conf > 0.5:  # Solo consideramos detecciones con alta confianza
                    detections.append((x_min, y_min, x_max, y_max, conf))
            if len(data) >= 4:  # Si el modelo devuelve coordenadas [x_min, y_min, x_max, y_max]
                x_min, y_min, x_max, y_max = (data * 300).astype(int)  # Escalar a 300x300
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, f"({x_min},{y_min})", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Mostrar imagen con detección
        cv2.imshow("OAK-1 Lite Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()



# import depthai as dai
# import numpy as np
# import cv2

# # Path to your custom model blob
# MODEL_BLOB_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/modelo.blob"

# # Create a pipeline
# pipeline = dai.Pipeline()

# # Create a camera node
# cam_rgb = pipeline.create(dai.node.ColorCamera)
# cam_rgb.setPreviewSize(300, 300)  # Change to match your model input size
# cam_rgb.setInterleaved(False)
# cam_rgb.setFps(30)

# # Create neural network node
# nn = pipeline.create(dai.node.NeuralNetwork)
# nn.setBlobPath(MODEL_BLOB_PATH)

# # Linking camera output to the neural network input
# cam_rgb.preview.link(nn.input)

# # Create output nodes
# xout_rgb = pipeline.create(dai.node.XLinkOut)
# xout_rgb.setStreamName("video")
# cam_rgb.preview.link(xout_rgb.input)

# xout_nn = pipeline.create(dai.node.XLinkOut)
# xout_nn.setStreamName("nn")
# nn.out.link(xout_nn.input)

# # Connect to device and start pipeline
# with dai.Device(pipeline) as device:
#     video_queue = device.getOutputQueue("video", maxSize=1, blocking=False)
#     nn_queue = device.getOutputQueue("nn", maxSize=1, blocking=False)

#     while True:
#         # Get video frame
#         frame = video_queue.get().getCvFrame()
#         frame = cv2.imread("C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/test.jpeg")
#         # Get inference results
#         in_nn = nn_queue.tryGet()
#         if in_nn:
#             data = np.array(in_nn.getFirstLayerFp16())  # Assuming float16 output
#             print("NN Output:", data)  # Process this according to your model

#         # Show frame
#         cv2.imshow("OAK-1 Lite", frame)

#         if cv2.waitKey(1) == ord('q'):
#             break

#     cv2.destroyAllWindows()
