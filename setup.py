"""
Crear un paquete de Python que contenga el proyecto de aprendizaje automático.
"""
# ------------------------------------------------------------------------------------------------
# Importar el módulo setuptools para crear un paquete de Python
import setuptools

# Abrir el archivo README.md en modo de lectura y codificación utf-8
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# ------------------------------------------------------------------------------------------------

__version__ = "0.0.0" # Definir la variable version con el valor “0.0.0”
REPO_NAME = "End-to-end-ML-Project-HousingPricesPrediction" # Definir la variable REPO_NAME con el nombre del repositorio de GitHub
AUTHOR_USER_NAME = "Ael-Dev" # autor del repositorio
SRC_REPO = "mlProject"
AUTHOR_EMAIL = "admi.alex.itz@gmail.com"

# ------------------------------------------------------------------------------------------------

# Llamar a la función setup del módulo setuptools para configurar el paquete
setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
# ------------------------------------------------------------------------------------------------
