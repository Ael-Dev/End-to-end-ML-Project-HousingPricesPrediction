import os
import sys
from mlProject import logger
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
import dill
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlProject.entity.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config


    def get_data_transformer_object(self,numerical_columns, categorical_columns):
        '''
            This function is responsible for data transformation
        '''
        try:
            
            # Define Pipeline
            # for numerical columns
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            # for categorical columns
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # logging info
            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")
            
            # Combination of pipelines
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise e


    def initiate_data_transformation(self):
        try:
            data_df = pd.read_csv(self.config.data_path)
            logger.info("Read dataset completed")

            # ----------------------------------------------------------------
            # define categorical, numerical columns and target column
            numerical_columns = ["area", "bedrooms","bathrooms","stories","parking"]
            categorical_columns = ["mainroad","guestroom","basement", "hotwaterheating","airconditioning","prefarea","furnishingstatus"]
            target_column_name = "price"

            # ----------------------------------------------------------------
            logger.info("Creating preprocessing object")
            preprocessing_obj = self.get_data_transformer_object(numerical_columns, categorical_columns)

            # ----------------------------------------------------------------
            # define variable X and y for training data
            input_feature_df=data_df.drop(columns=[target_column_name],axis=1)
            target_feature_df=data_df[target_column_name]

            # ----------------------------------------------------------------
            logger.info(f"Applying preprocessing object on training and testing dataframe")
            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)

            # ----------------------------------------------------------------

            # concatenate preprocessed input features with target feature
            data_arr = np.c_[input_feature_arr, np.array(target_feature_df)]
            # define the column names as a list
            column_names = list(preprocessing_obj.get_feature_names_out()) + [target_column_name]
            # removing prefixes of columns names
            column_names = [name.replace("cat_pipelines__", "").replace("num_pipeline__", "") for name in column_names]
            # convert the array to a dataframe
            data_df = pd.DataFrame(data_arr, columns=column_names)


            # ----------------------------------------------------------------
            # Split the data into training and test sets. (0.75, 0.25) split.
            train, test = train_test_split(data_df)#data_arr)
            logger.info("Splited data into training and test sets")

            logger.info(train.shape)
            logger.info(test.shape)

            # ----------------------------------------------------------------
            logger.info(f"saving the pre-processing data")
            train.to_csv(os.path.join(self.config.root_dir, "train.csv"),index = False)
            test.to_csv(os.path.join(self.config.root_dir, "test.csv"),index = False)

            
            # ----------------------------------------------------------------
            # save object
            ruta = self.config.root_dir + '/preprocessing_obj.pkl'
            with open(ruta, 'wb') as f:
                pickle.dump(preprocessing_obj, f)
                
            logger.info(f"Saved preprocessing object")

        except Exception as e:
            raise e
        