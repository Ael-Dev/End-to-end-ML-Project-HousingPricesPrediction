import os
from pathlib import Path
import logging

# ------------------------------------------------------------------------------------------------
# configuracion basica para el loggin del sistema

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# ------------------------------------------------------------------------------------------------
# nombre del proyecto
project_name = "mlProject"

# estructura de carpetas y archivos
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "notebooks/trials.ipynb",
    "templates/index.html",
    "test.py"
]

# ------------------------------------------------------------------------------------------------

# recorrer una lista de archivos
for filepath in list_of_files:
    # convertir la ruta del archivo en un objeto Path
    filepath = Path(filepath) 
    # separar la ruta del archivo en el directorio y el nombre del archivo filedir
    filedir, filename = os.path.split(filepath)
    # verificar si el directorio no está vacío
    if filedir !="":
        # crear el directorio si no existe
        os.makedirs(filedir, exist_ok=True)
        # registrar un mensaje informativo sobre la creación del directorio
        logging.info(f"Creating directory; {filedir} for the file: {filename}")
    # verificar si el archivo no existe o está vacío
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        # abrir el archivo en modo de escritura
        with open(filepath, "w") as f:
            pass
            # registrar un mensaje informativo sobre la creación del archivo vacío 
            logging.info(f"Creating empty file: {filepath}")
    # si el archivo ya existe y no está vacío
    else:
        # registrar un mensaje informativo sobre la existencia del archivo
        logging.info(f"{filename} is already exists")
# ------------------------------------------------------------------------------------------------
