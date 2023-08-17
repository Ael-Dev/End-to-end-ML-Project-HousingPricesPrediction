import os
import sys
import logging

"""
configurar el módulo logging para registrar los mensajes de información, 
advertencia, error y otros niveles sobre el funcionamiento 
de un proyecto de aprendizaje automático.
"""
# ------------------------------------------------------------------------------------------------
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]" # Definir la variable logging_str con el formato de los mensajes de registro
log_dir = "logs" # Definir la variable log_dir con el nombre del directorio donde se guardarán los archivos de registro
log_filepath = os.path.join(log_dir,"running_logs.log") # Definir la variable log_filepath con la ruta del archivo de registro
os.makedirs(log_dir, exist_ok=True) # Crear el directorio de registro si no existe
# ------------------------------------------------------------------------------------------------
# Configurar el módulo logging con los siguientes parámetros:
logging.basicConfig( 
    level= logging.INFO, # Establecer el nivel de registro como INFO
    format= logging_str, # Establecer el formato de los mensajes de registro
    # Establecer los manejadores de registro como un archivo y una salida estándar
    handlers=[
        logging.FileHandler(log_filepath), # Crear un manejador de archivo
        logging.StreamHandler(sys.stdout) # Crear un manejador de salida estándar
    ]
)
# ------------------------------------------------------------------------------------------------
# Obtener un registrador con el nombre “mlProjectLogger”
logger = logging.getLogger("mlProjectLogger")
# ------------------------------------------------------------------------------------------------
