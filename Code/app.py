import sys
import depthai as dai
from gui import MyWindow

import cv2
import qdarktheme
#immportar la libreria
from importlib.metadata import version, metadata
print("Versión lib_pose:", version("lib_pose"))
# meta = metadata("lib_pose")
# print("Autor:", meta["Author"])
# print("Descripción:", meta["Summary"])



from lib_pose import Pose
oPose = Pose.cPose()


# Import personal class
from cCamera import *

from cLogger import cLogger

import time
import datetime
from PyQt5.QtWidgets import *

from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtCore import Qt



class cWork(QThread):
    imageupd = pyqtSignal(QImage)
    listResultupd = pyqtSignal(bool)

    def __init__(self, olog, ocamera, parent = None):
        super().__init__(parent)
        self.ocamera = ocamera
        self.olog = olog
        print("Instancia de cWork creada")
        self.detectar_activo = False  # Variable interna

    def run (self):
        self.olog.logger.debug("Empezou fio")
        self.thread_running = True

        c =0
        # Var to calculate frame time
        prev_frame_time = 0

        while self.thread_running:
            try:
                self.ocamera.bshow_box =  self.detectar_activo
                frame = self.ocamera.capture()
                #Test mask
                #frame = cv2.rectangle(frame,(100,120),(295,500),(0,0,0),-1)
                frame = cv2.rectangle(frame,(295,120),(450,500),(0,0,0),-1)

                listaPersonas= None
                signal = None
                if frame is not None:
                    self.latest_frame = frame.copy()
                    #use lib
                    frame, listaPersonas, signal = oPose.inference(frame, c)
                    print(listaPersonas)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if not frame.flags['C_CONTIGUOUS']:
                        frame = np.ascontiguousarray(frame)
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()

                    c=c+1
                    print(c)
                    # Calculate frames
                    current_time = time.time()
                    try:
                        fps = 1 / (current_time - prev_frame_time)
                    except ZeroDivisionError:
                        fps =0
                    prev_frame_time = current_time

                    cv2.putText(frame, "FPS: %.0f" % fps, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)

                    self.imageupd.emit(qimg)
                    self.listResultupd.emit(signal)
                    time.sleep(0.01)
            except Exception as e:
                print(f"error while {e}")

    def get_latest_frame(self):
        return self.latest_frame

    def set_detectar_activo(self, state: bool):
        self.detectar_activo = state



    # def cancel(self):
    #     self.label.clear()
    #     self.stop()

    def stop (self):
        self.thread_running = False

        self.quit()
        self.wait()


class CameraViewer(QWidget):

    _rbActivateDetections_state_changed = pyqtSignal(bool)

    def __init__(self, window):
        super().__init__()
        self.window = window
        print("Creando instancia de MiClase")

        #self.init_gui()

        #Init Logger
        self.olog = cLogger("app_log")

        self.olog.logger.info("Init app")



        #self.capture = cv2.VideoCapture(0)
        self.list_init_cameras =[0,0,0,0]



    # def init_gui(self,):

    #     self.setWindowTitle("Camera viewer")
    #     self.setGeometry(100, 100, 120, 900)

    #     # Widgets principales
    #     self.video_label = QLabel()
    #     self.video_label.setFixedSize(1024, 768)

    #     # Controles a la derecha
    #     self.combobox = QComboBox()
    #     self.combobox.addItems(["Oak1", "Hikvision","Webcam"])

    #     self.boton1 = QPushButton("Start")
    #     self.boton2 = QPushButton("stop")
    #     self._rbActivateDetections = QRadioButton ("Show detections")
    #     self._bCapture = QPushButton("Captura")
    #     self.atras = QPushButton("< ")
    #     self.adelante = QPushButton(">")
    #     self.lResult = QLabel()
    #     # Establecer tamaño de fuente
    #     font = QFont("Arial", 16)  # Fuente Arial, tamaño 16
    #     self.lResult.setFont(font)

    #     # Establecer tamaño fijo (ancho x alto)
    #     self.lResult.setFixedSize(300, 50)

    #     # Alinear texto al centro si quieres
    #     self.lResult.setAlignment(Qt.AlignCenter)

    #     self._bCapture.setEnabled(False)

    #     self.textbox = QLineEdit()
    #     self.textbox.setPlaceholderText("Escribe algo...")

    #     # Layout derecho
    #     side_layout = QVBoxLayout()
    #     side_layout.addWidget(self.combobox)
    #     side_layout.addWidget(self.boton1)
    #     side_layout.addWidget(self.boton2)

    #     side_layout.addWidget(self._rbActivateDetections)
    #     side_layout.addWidget(self.textbox)
    #     side_layout.addWidget(self._bCapture)
    #     side_layout.addWidget(self.atras)
    #     side_layout.addWidget(self.adelante)
    #     side_layout.addWidget(self.lResult)

    #     side_layout.addStretch()

    #     # Layout principal
    #     main_layout = QHBoxLayout()
    #     main_layout.addWidget(self.video_label)
    #     main_layout.addLayout(side_layout)
    #     self.setLayout(main_layout)


    #     #Eventos
    #     self.boton1.clicked.connect(self.start_video)
    #     self.boton2.clicked.connect (self.parar)
    #     self._rbActivateDetections.toggled.connect(self.on_radio_changed)
    #     self._bCapture.clicked.connect(self.on_capturar_imagen)
    #     # self.atras.clicked.connect(self.show_image)


    #     #para ver imagenes

    # def listar_imagenes  (self,):

    # def show_image(self,):


    def parar(self,):
        if self.work:
            self.work.stop()

    def imageupd_slot(self, image):
        if not image.isNull():
            # Escalar la imagen al tamaño del QLabel manteniendo la proporción
            image_escalado = image.scaled(self.window.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.window.video_label.setPixmap(QPixmap.fromImage(image_escalado))
            
    def listResultupd_slot(self,signal):
        self.window.lResult.setText(str(signal))
        seleccionar_color = lambda x: "background-color: lightgreen" if x == True else "background-color: red"  
        self.window.lResult.setStyleSheet(seleccionar_color(signal))

       # self.video_label.setPixmap(QPixmap.fromImage(image))

    def selection_camera(self,):
        camera = None

        selected_camera = self.window.combobox.currentIndex()
        if selected_camera == 0:
            try:
                if self.list_init_cameras[selected_camera] == 0:
                    # Create pipeline
                    pipeline = dai.Pipeline()
                    # Cámara
                    self.oCameraOak = cCamera(self.olog, pipeline, 320, 320)
                    self.oCameraOak.connect()
                    self.list_init_cameras[selected_camera] = 1
                camera = self.oCameraOak


            except Exception as e:
                self.olog.logger.warn(f"Error to connect camera OAK: ")
        elif selected_camera ==1:
            try:
                if self.list_init_cameras[selected_camera] == 0:
                    self.oipcamera = cIpCamera(self.olog)
                    self.oipcamera.connect()
                    print (self.oipcamera)
                    self.list_init_cameras[selected_camera] = 1
                camera = self.oipcamera

            except Exception:
                self.olog.logger.warn("error conect camera ip")
        elif selected_camera == 2:
            try:
                if self.list_init_cameras[selected_camera] == 0:
                    self.ocameraweb = cWebcam(self.olog)

                    self.ocameraweb.connect()
                    self.list_init_cameras[selected_camera] = 1
                camera = self.ocameraweb

            except Exception:
                self.olog.logger.warn("error conect camera web")

        elif selected_camera == 3:
            try:
                if self.list_init_cameras[selected_camera] == 0:
                    self.oDummycam = cDummyCam(self.olog)

                    self.oDummycam.connect()
                    self.list_init_cameras[selected_camera] = 1
                camera = self.oDummycam

            except Exception:
                self.olog.logger.warn("error conect DummyCam")

        return camera

    def start_video (self,):
        current_camera = self.selection_camera()
        if not  current_camera.bIsConnected:
            current_camera.reconnect()
        print(">> Lanzando hilo")
        self.work = cWork(self.olog, current_camera)
        self.work.imageupd.connect(self.imageupd_slot)
        self.work.listResultupd.connect(self.listResultupd_slot)
        self._rbActivateDetections_state_changed.connect(self.work.set_detectar_activo)
        self.work.start()
        self.window._bCapture.setEnabled(True)

    #
    def on_radio_changed(self, checked):
        self._rbActivateDetections_state_changed.emit(checked)

    def on_capturar_imagen(self):
        frame = self.work.get_latest_frame()
        if frame is not None:
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S.%f")
            filename = f"{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            self.olog.logger.debug(f"Imagen guardada como {filename}")
        else:
            self.olog.logger.debug("No hay imagen para capturar.")

    #Inernal events
    def closeEvent(self, event ):
        # Ask for confirmation before closing
        confirmation = QMessageBox.question(self, "Confirmation", "Are you sure you want to close the application?", QMessageBox.Yes | QMessageBox.No)

        if confirmation == QMessageBox.Yes:
            self.olog.logger.info("Close app!")
            event.accept()  # Close the app
        else:
            event.ignore()  # Don't close the app


    # def closeEvent(self, event):
    #     self.capture.release()

def main():
    app = QApplication(sys.argv)

    window = MyWindow ()

    viewer = CameraViewer(window)

    #Eventos
    window.boton1.clicked.connect(viewer.start_video)
    window.boton2.clicked.connect (viewer.parar)
    window._rbActivateDetections.toggled.connect(viewer.on_radio_changed)
    window._bCapture.clicked.connect(viewer.on_capturar_imagen)



    window.show()



    qdarktheme.setup_theme()
    # palette = QPalette()
    # palette.setColor(QPalette.Window, QColor(30, 30, 30))  # Dark gray background
    # palette.setColor(QPalette.WindowText, QColor(255, 255, 255))  # White text
    # app.setPalette(palette)


   # viewer.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()