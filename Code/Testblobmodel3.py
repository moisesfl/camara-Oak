# first, import all necessary modules
from pathlib import Path

import blobconverter
import cv2
import depthai
import numpy as np
import time

pipeline = depthai.Pipeline()


img_filename = "C:/Users/moifra3484/Desktop/Proyectos/MCU/crearmodelo/my_model/test.jpeg"

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(320, 320)  # 300x300 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

detection_nn = pipeline.createMobileNetDetectionNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
#detection_nn.setBlobPath("C:/Users/moifra3484/Downloads/result2/my_model_openvino_2022.1_6shave.blob")
detection_nn.setBlobPath("C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/my_model-simplified.blob")


# Next, we filter out the detections that are below a confidence threshold. Confidence can be anywhere between <0..1>
detection_nn.setConfidenceThreshold(0.1)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")

cam_rgb.preview.link(xout_rgb.input)
cam_rgb.preview.link(detection_nn.input)
detection_nn.out.link(xout_nn.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    detections = []

    # Since the detections returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        #frame = cv2.imread(img_filename)
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    c = 1
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()
            #frame = cv2.imread("C:/Users/moifra3484/Desktop/Proyectos/MCU/modelo_boli/my_model/test.jpg")
            #frame = cv2.imread(img_filename)

        if in_nn is not None:
            # when data from nn is received, we take the detections array that contains mobilenet-ssd results
            detections = in_nn.detections


        if frame is not None:
            for detection in detections:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord('s'):
            ##test:delete
            # print("⏳ Esperando 3 segundos...")
            # time.sleep(5)  # Espera 3 segundos antes de capturar la imagen
            # frame = in_rgb.getCvFrame()
            ##        
            cv2.imwrite("cap/capturatest_"+ str(c) + ".jpg", frame)
            c = c+1
        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            break


