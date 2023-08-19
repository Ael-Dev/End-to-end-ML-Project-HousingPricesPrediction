from mlProject.constants import *
from mlProject.utils.common import read_yaml, create_directories 
from mlProject.entity.config_entity import DataIngestionConfig, DataTransformationConfig, DataValidationConfig 

class ConfigurationManager:

   # ------------------------------------------------------------------------------------------------
  
    # Definir el método constructor que recibe las rutas de los archivos de configuración, parámetros y esquema
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        # Leer los archivos YAML y asignarlos a los atributos de la instancia
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        # Crear el directorio raíz para los artefactos del proyecto
        create_directories([self.config.artifacts_root])

  # ------------------------------------------------------------------------------------------------

    # Definir un método que devuelve la configuración para la ingesta de datos
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        # Obtener la configuración de la ingesta de datos
        config = self.config.data_ingestion 

        # Crear el directorio raíz para la ingesta de datos
        create_directories([config.root_dir])
        # Crear un objeto de la clase DataIngestionConfig con los parámetros(paths) de la configuración
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    # ------------------------------------------------------------------------------------------------

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS # obtener las columnas desde el archivo yaml
        # Crear el directorio raíz para la validación de datos
        create_directories([config.root_dir])
        # Crear un objeto de la clase DataValidationConfig con los parámetros de la configuración
        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir = config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config
    

    # ------------------------------------------------------------------------------------------------

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config