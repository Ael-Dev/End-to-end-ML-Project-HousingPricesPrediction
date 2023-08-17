import os
import urllib.request as request
import zipfile
from mlProject import logger
from mlProject.utils.common import get_size

from pathlib import Path
from mlProject.entity.config_entity import (DataIngestionConfig)


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    # Definir un método que descarga el archivo de datos de la URL de la fuente
    def download_file(self):
        # Verificar si el archivo local ya existe
        if not os.path.exists(self.config.local_data_file):  
            # Si no existe, descargar el archivo de la URL y guardar el nombre y los encabezados
            filename, headers = request.urlretrieve(
                url = self.config.source_URL,
                filename = self.config.local_data_file
            ) 
            # Registrar un mensaje con el nombre del archivo y los encabezados
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")



    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        # Asignar la ruta del directorio donde se va a descomprimir el archivo
        unzip_path = self.config.unzip_dir 
        # Crear el directorio si no existe
        os.makedirs(unzip_path, exist_ok=True)
        # Abrir el archivo zip en modo lectura
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            # Extraer todos los archivos del zip al directorio
            zip_ref.extractall(unzip_path)
  