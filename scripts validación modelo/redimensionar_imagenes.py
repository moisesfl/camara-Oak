import os
import cv2

def redimensionar_imagenes_cv2(origen, destino, nuevo_ancho, nuevo_alto):
    # Crear carpeta destino si no existe
    os.makedirs(destino, exist_ok=True)

    for archivo in os.listdir(origen):
        ruta_origen = os.path.join(origen, archivo)
        ruta_destino = os.path.join(destino, archivo)

        # Leer la imagen
        imagen = cv2.imread(ruta_origen)
        if imagen is None:
            print(f"No se pudo leer la imagen: {archivo}")
            continue

        # Redimensionar imagen
        imagen_redimensionada = cv2.resize(imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

        # Guardar en carpeta destino
        cv2.imwrite(ruta_destino, imagen_redimensionada)
        print(f"Procesada: {archivo}")

# Ejemplo de uso
carpeta_origen = "dataset_3_02-05-may-2025"
carpeta_destino = "imagenes_redimensionadas_dataset_3"
ancho = 320
alto = 320

redimensionar_imagenes_cv2(carpeta_origen, carpeta_destino, ancho, alto)
