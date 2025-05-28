#Cargar blobl model y ver en tiempo real la detección de objetos
configPath = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model.json"
#!/usr/bin/env python3
#!/usr/bin/env python3
from abc import ABC, abstractmethod
from cLogger import cLogger

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import json
import datetime

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

# olog = cLogger("app_log")
# olog.p()
# olog.logger.error("sdf")


# Get argument first
# nnPath = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model_openvino_2022.1_6shave.blob"
# CONFIG_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model.json"

# cfg = ModelConfig(CONFIG_PATH)
# print (cfg.__repr__)

# if not Path(nnPath).exists():
#     import sys
#     raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# # tiny yolo v4 label texts
# labelMap = [
#     "boli"
# ]
# #También coger las clases del archivo .json
# labelMap = cfg.labels


# syncNN = True

# # Create pipeline
# pipeline = dai.Pipeline()
class cModelBlob():
   def __init__(self,oLog, pipeline):
       try:
        blob_model_path = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model_openvino_2022.1_6shave.blob" 
        blob_model_config_path = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model.json"

        if not Path(blob_model_path).exists():
            raise FileNotFoundError(f'Required file/s not found: {blob_model_path}')
        cfg = ModelConfig(blob_model_config_path)
        self.labelmap = cfg.labels
        print (self.labelmap)
        self.detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
        nnOut = pipeline.create(dai.node.XLinkOut)
        nnOut.setStreamName("nn")

        # # Network specific settings
        self.detectionNetwork.setConfidenceThreshold(0.5)
        self.detectionNetwork.setNumClasses(1)
        self.detectionNetwork.setNumClasses(cfg.num_classes) #get from .json
        self.detectionNetwork.setCoordinateSize(4)
        self.detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
        self.detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
        self.detectionNetwork.setIouThreshold(0.3)
        self.detectionNetwork.setBlobPath(blob_model_path)
        self.detectionNetwork.setNumInferenceThreads(2)
        self.detectionNetwork.input.setBlocking(False)


        self.detectionNetwork.out.link(nnOut.input)
       except Exception:
           oLog.logger.debug("error: ")

       

class CameraInterface(ABC):
    bIsConnected = False

    @abstractmethod
    def connect(self):
        """Método obligatorio para conectar la cámara"""
        pass

    @abstractmethod
    def capture(self):
        """Método obligatorio para conectar la cámara"""
        pass

class cWebcam(CameraInterface):
    def __init__(self, olog):
        self.bIsConnected= False
        self.olog = olog
        
    def connect(self):
        try:
            self.cap = cv2.VideoCapture(0)
            self.bIsConnected= True
        except Exception:
            print("error al conectar")
            self.bIsConnected = False

    def reconnect(self,):
        if not self.bIsConnected:
            self.cap = cv2.VideoCapture(0)
            self.bIsConnected= True

    def capture(self):
        ret, frame = self.cap.read()
        return frame
class cDummyCam (CameraInterface):
    def __init__(self, olog):
        self.bIsConnected= False
        self.olog = olog
        self.read_video_path = 'C:/Users/moifra3484/OneDrive - CTAG/TECNOVE/modelo machine learning/Dataset1/output1.mp4'
        self.read_video_path = 'C:/Users/moifra3484/OneDrive - CTAG/TECNOVE/modelo machine learning/Dataset3/3.3.mp4'




    def connect(self):
        try:
            self.cap = cv2.VideoCapture(self.read_video_path)
            self.bIsConnected= True
        except Exception:
            print("error al conectar")
            self.bIsConnected = False
    def reconnect(self,):
        if not self.bIsConnected:
            self.cap = cv2.VideoCapture(self.read_video_path)
            self.bIsConnected= True

    def capture(self):
        ret, frame = self.cap.read()
        return frame
          
class cIpCamera(CameraInterface):

    def __init__(self, olog):
        # Dirección RTSP de tu cámara IP (ajusta usuario, contraseña y IP)
        self.url = 'rtsp://admin:Ctag_2025@192.168.1.64:554/Streaming/Channels/102'
        self.bIsConnected= False

        self.olog = olog

    def connect(self):
        try:
            # Abre el stream
            self.cap = cv2.VideoCapture(self.url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) #
            self.bIsConnected= True

            if not self.cap.isOpened():
                self.olog.logger.warn("No se pudo conectar al stream")
        except Exception:
            print("error al conectar")
            self.bIsConnected = False

    def reconnect(self,):
        if not self.bIsConnected:
           self.cap = cv2.VideoCapture(self.url)
           self.bIsConnected= True

    def read_latest_frame(self, discard_count=5):
        for _ in range(discard_count):
            self.cap.grab()  # Desecha los frames anteriores
        ret, frame = self.cap.read()
        return frame if ret else None

    def capture(self):
        #ret, frame = self.cap.read()
        frame = self.read_latest_frame()
        return frame

class cCamera(CameraInterface):
    device = None

    def __init__(self, olog, pipeline, width = 320, height = 320):
        self.pipeline = pipeline
        self.camRgb = pipeline.create(dai.node.ColorCamera)
        self.width = width
        self.height = height
        self.Properties()
        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")

        self.bshow_box = False
        self.bActivate_blob_model = True
        self.oModelBlob = cModelBlob(olog, pipeline)

        # Linking
        if self.bActivate_blob_model:
            self.camRgb.preview.link(self.oModelBlob.detectionNetwork.input)
            self.oModelBlob.detectionNetwork.passthrough.link(xoutRgb.input)
        else:
            self.camRgb.preview.link(xoutRgb.input)      

        self.bIsConnected= False


    def Properties(self,):
        self.camRgb.setPreviewSize(self.width, self.height)
        self.camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        self.camRgb.setInterleaved(False)
        self.camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        self.camRgb.setFps(40)

    def connect(self,):
        try:
            self.device = dai.Device(self.pipeline)
            self.bIsConnected= True
        except Exception:
            print("error al conectar")
            self.bIsConnected = False

    def reconnect(self,):
        if not self.bIsConnected:
           self.device = dai.Device(self.pipeline)
           self.bIsConnected= True

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(self,frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(self, frame, detections):
        
        color = (255, 0, 0)
        for detection in detections:
            bbox = self.frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, self.oModelBlob.labelmap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        return frame


    def capture (self,):
            frame = None
            qDet =None
            inDet = None
            qRgb = None
            detections = []

            if self.device is not None:
                # Output queues will be used to get the rgb frames and nn data from the outputs defined above
                qRgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
                if self.bActivate_blob_model:                 
                    qDet = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)

            if self.bActivate_blob_model:

                inRgb = qRgb.get()
                inDet = qDet.get()
            else:
                inRgb = qRgb.tryGet()
  

            if inRgb is not None:
                frame = inRgb.getCvFrame()

            if inDet is not None:
                detections = inDet.detections
            if frame is not None and self.bshow_box:
               frame = self.displayFrame(frame, detections)
            return frame
                    



# ocamera  = cCamera(pipeline,500,500)
# ocamera.connect()


# # Define sources and outputs
# camRgb = pipeline.create(dai.node.ColorCamera)
# detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
# xoutRgb = pipeline.create(dai.node.XLinkOut)
# nnOut = pipeline.create(dai.node.XLinkOut)

# xoutRgb.setStreamName("rgb")
# nnOut.setStreamName("nn")

# # Properties
# camRgb.setPreviewSize(320, 320)
# camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
# camRgb.setInterleaved(False)
# camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
# camRgb.setFps(40)

# # Network specific settings
# detectionNetwork.setConfidenceThreshold(0.5)
# detectionNetwork.setNumClasses(1)
# detectionNetwork.setNumClasses(cfg.num_classes) #get from .json
# detectionNetwork.setCoordinateSize(4)
# detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
# detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
# detectionNetwork.setIouThreshold(0.3)
# detectionNetwork.setBlobPath(nnPath)
# detectionNetwork.setNumInferenceThreads(2)
# detectionNetwork.input.setBlocking(False)

# # Linking
# camRgb.preview.link(detectionNetwork.input)
# if syncNN:
#     detectionNetwork.passthrough.link(xoutRgb.input)
# else:
#     camRgb.preview.link(xoutRgb.input)

# detectionNetwork.out.link(nnOut.input)

# # Connect to device and start pipeline
# with dai.Device(pipeline) as device:

#     # Output queues will be used to get the rgb frames and nn data from the outputs defined above
#     qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
#     qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

#     frame = None
#     detections = []
#     startTime = time.monotonic()
#     counter = 0
#     color2 = (255, 255, 255)

#     # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
#     def frameNorm(frame, bbox):
#         normVals = np.full(len(bbox), frame.shape[0])
#         normVals[::2] = frame.shape[1]
#         return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

#     def displayFrame(name, frame):
#         color = (255, 0, 0)
#         for detection in detections:
#             bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
#             cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
#             cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
#             cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
#         # Show the frame
#         cv2.imshow(name, frame)

#     while True:
#         if syncNN:
#             inRgb = qRgb.get()
#             inDet = qDet.get()
#         else:
#             inRgb = qRgb.tryGet()
#             inDet = qDet.tryGet()

#         if inRgb is not None:
#             frame = inRgb.getCvFrame()
#             cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
#                         (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

#         if inDet is not None:
#             detections = inDet.detections
#             counter += 1

#         if frame is not None:
#             displayFrame("rgb", frame)

#         if cv2.waitKey(1) == ord('q'):
#             break

