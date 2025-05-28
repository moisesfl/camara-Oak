import cv2, time
import depthai as dai

# Crear una pipeline
pipeline = dai.Pipeline()

# Crear un nodo de cámara color
cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Crear un nodo de salida XLink
xout_video = pipeline.create(dai.node.XLinkOut)
xout_video.setStreamName("video")

# Conectar la salida de la cámara al nodo de salida
cam_rgb.video.link(xout_video.input)

# Iniciar la conexión con el dispositivo
with dai.Device(pipeline) as device:
    video_queue = device.getOutputQueue(name="video", maxSize=1, blocking=False)
    c=1
    while True:
        frame = video_queue.get().getCvFrame()  # Obtener el frame
        cv2.imshow("OAK-1 Lite Live Stream", frame)  # Mostrar la imagen

        key = cv2.waitKey(1)
        if key == ord('s'):  # Press 's' to capture
            print("Image Captured!")
            time.sleep(4)
            frame = video_queue.get().getCvFrame()  # Obtener el frame
            cv2.imwrite(f"captured_image_{c}.jpg", frame)
            c = c+1
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Presiona 'q' para salir
            break

cv2.destroyAllWindows()
