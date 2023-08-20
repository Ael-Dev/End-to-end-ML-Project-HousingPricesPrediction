import os
import sys
from dataclasses import dataclass

import pandas as pd
from mlProject import logger
import joblib

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV
from mlProject.entity.config_entity import (ModelTrainerConfig)

# Crear una clase para entrenar el modelo
class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, param):
        try:
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i] # Retrieve the model
                # -------------------------------------------------------
                # hyperparameter params
                para=param[list(models.keys())[i]]
                gs = GridSearchCV(model,para,cv=3)
                gs.fit(X_train, y_train) # Fit the model

                # Retrieve the best params
                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train) # fit with the best params
                # -------------------------------------------------------
                # Get the predictions
                y_train_pred = model.predict(X_train) # train
                y_test_pred = model.predict(X_test) # test
                # Get the scores
                train_model_score = r2_score(y_train, y_train_pred)
                test_model_score = r2_score(y_test, y_test_pred)
                # Make the report
                report[list(models.keys())[i]] = test_model_score
            
            return report

        except Exception as e:
            raise e
    
    def train(self):
        # Cargar el dataset
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
        logger.info("Read dataset completed")

        # establer la variable "X" y "y" tanto en el dataset de entrenamient y test
        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        # Split the dataset into train and test
        #logging.info("Split trining and test input data")
        X_train,y_train,X_test,y_test=(
            train_x.iloc[:,:-1],
            train_y.iloc[:,-1],
            test_x.iloc[:,:-1],
            test_y.iloc[:,-1]
        )
        logger.info("Splitted dataset completed")

        # Initialize the models
        models = {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }

        # Hyperparameter Tunning
        params={
            "Decision Tree": {
                'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                'learning_rate':[.1,.01,.05,.001],
                'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Linear Regression":{},
            "XGBRegressor":{
                'learning_rate':[.1,.01,.05,.001],
                'n_estimators': [8,16,32,64,128,256]
            },
            "CatBoosting Regressor":{
                'depth': [6,8,10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor":{
                'learning_rate':[.1,.01,0.5,.001],
                # 'loss':['linear','square','exponential'],
                'n_estimators': [8,16,32,64,128,256]
            }   
        }


        # Evaluate the models
        model_report:dict=self.evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                            models=models, param=params)
        logger.info("models trained")


        # Get the best model score from dictionary
        best_model_score = max(sorted(model_report.values()))
        # Get best model name from dictionary
        index_best_model = list(model_report.values()).index(best_model_score)
        best_model_name = list(model_report.keys())[index_best_model]
        # Get the best model
        best_model = models[best_model_name]
        

        # limit the model score
        threshold = 0.6 # limit
        if best_model_score < threshold:
            print("No best model found!")
        logger.info("best model obtained")
        #logging.info(f"Best found model on both training and testing dataset {best_model_name}")
        
        # save best model
        logger.info("saving the best model")
        # Guardar el modelo entrenado en la ruta establecida(dentro de artifacts)
        joblib.dump(best_model, os.path.join(self.config.root_dir, self.config.model_name))
# ------------------------------------------------------------------------------------------------