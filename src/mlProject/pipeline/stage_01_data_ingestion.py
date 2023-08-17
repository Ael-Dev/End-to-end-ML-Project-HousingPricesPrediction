from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_ingestion import DataIngestion
from mlProject import logger

STAGE_NAME = "Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Realizar la ingesta de datos desde una fuente remota 
        a un directorio local, usando las clases ConfigurationManager y DataIngestion
        """
        config = ConfigurationManager() # Crear un objeto de la clase ConfigurationManager que gestiona la configuración del proyecto 
        data_ingestion_config = config.get_data_ingestion_config() # Obtener la configuración para la ingesta de datos 
        data_ingestion = DataIngestion(config=data_ingestion_config) # Crear un objeto de la clase DataIngestion que realiza la ingesta de datos usando la configuración obtenida
        data_ingestion.download_file() # Descargar el archivo de datos de la URL
        data_ingestion.extract_zip_file() # Descomprimir el archivo de datos en el directorio local 

if __name__ == "__main__":
    try:        
        logger.info(f"====>>> stage {STAGE_NAME} started <<<=====")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"====>>> stage {STAGE_NAME} completed <<<=====\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
    