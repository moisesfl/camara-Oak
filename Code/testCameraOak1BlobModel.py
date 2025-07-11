#Cargar blobl model y ver en tiempo real la detección de objetos
configPath = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model.json"
#!/usr/bin/env python3
#!/usr/bin/env python3

from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
import json

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

# Get argument first
nnPath = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model_openvino_2022.1_6shave.blob"
CONFIG_PATH = "C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/result (2)/my_model.json"

cfg = ModelConfig(CONFIG_PATH)
print (cfg.__repr__)

if not Path(nnPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# tiny yolo v4 label texts
labelMap = [
    "boli"
]
#También coger las clases del archivo .json
labelMap = cfg.labels


syncNN = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Properties
camRgb.setPreviewSize(320, 320)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network specific settings
detectionNetwork.setConfidenceThreshold(0.5)
detectionNetwork.setNumClasses(1)
detectionNetwork.setNumClasses(cfg.num_classes) #get from .json
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319])
detectionNetwork.setAnchorMasks({"side26": [1, 2, 3], "side13": [3, 4, 5]})
detectionNetwork.setIouThreshold(0.3)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        # Show the frame
        cv2.imshow(name, frame)

    while True:
        if syncNN:
            inRgb = qRgb.get()
            inDet = qDet.get()
        else:
            inRgb = qRgb.tryGet()
            inDet = qDet.tryGet()

        if inRgb is not None:
            frame = inRgb.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)

        if inDet is not None:
            detections = inDet.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break

