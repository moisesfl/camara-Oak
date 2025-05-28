from PyQt5.QtWidgets import *

from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtCore import Qt


class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Camera viewer")
        self.setGeometry(100, 100, 120, 900)

        # Widgets principales
        self.video_label = QLabel()
        self.video_label.setFixedSize(1024, 768)
        self.video_label.setAlignment(Qt.AlignCenter)


        # Controles a la derecha
        self.combobox = QComboBox()
        self.combobox.addItems(["Oak1", "Hikvision","Webcam","DummyCam"])

        self.boton1 = QPushButton("Start")
        self.boton2 = QPushButton("stop")
        self._rbActivateDetections = QRadioButton ("Show detections")
        self._bCapture = QPushButton("Captura")
        self.atras = QPushButton("< ")
        self.adelante = QPushButton(">")
        self.lResultText = QLabel("Resultado:")
        self.lResult = QLabel()
        # Establecer tamaño de fuente
        font = QFont("Arial", 16)  # Fuente Arial, tamaño 16
        font.setBold(True)
        self.lResultText.setFont(font)
        self.lResult.setFont(font)

        # Establecer tamaño fijo (ancho x alto)
        self.lResult.setFixedSize(300, 50)

        # Alinear texto al centro si quieres
        self.lResultText.setAlignment(Qt.AlignCenter)
        self.lResult.setAlignment(Qt.AlignCenter)

        self._bCapture.setEnabled(False)

        self.textbox = QLineEdit()
        self.textbox.setPlaceholderText("Escribe algo...")

        # Layout derecho
        side_layout = QVBoxLayout()
        side_layout.addWidget(self.combobox)
        side_layout.addWidget(self.boton1)
        side_layout.addWidget(self.boton2)

        side_layout.addWidget(self._rbActivateDetections)
        side_layout.addWidget(self.textbox)
        side_layout.addWidget(self._bCapture)
        side_layout.addWidget(self.atras)
        side_layout.addWidget(self.adelante)
        side_layout.addWidget(self.lResultText)
        side_layout.addWidget(self.lResult)

        side_layout.addStretch()

        # Layout principal
        main_layout = QHBoxLayout()
        main_layout.addWidget(self.video_label)
        main_layout.addLayout(side_layout)
        self.setLayout(main_layout)


 

