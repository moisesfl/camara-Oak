import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import os

class cLogger():
    
    def __init__(self, name):
        self.name = name

        #Constant 
        folder_name_log = "log"      
         
        self.absolut_path_log = os.path.join (os.getcwd(), folder_name_log)
        self.create_folder()

        # Configurar el registro
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        file_name = datetime.now().strftime("%Y-%m-%d")
        handler_file = logging.handlers.RotatingFileHandler(f'{file_name}.log', maxBytes=1024, backupCount=3)
        handler_console =logging.StreamHandler()

        # Establecer el formato de los mensajes de registro
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler_file.setFormatter(formatter)
        handler_console.setFormatter(formatter)

        # Agregar el manejador de registro al objeto Logger
        self.logger.addHandler(handler_file)
        self.logger.addHandler(handler_console)



    def create_folder(self, ):
        if not Path(self.absolut_path_log).exists():
            os.makedirs(self.absolut_path_log)

    def p(self, ):
        print (self.name )