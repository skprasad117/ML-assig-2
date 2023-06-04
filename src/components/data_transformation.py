# ******************************************************************
# * Data Transformation : Main purpose is to do                    *
# *                       -> Data Cleaning                         *
# *                       -> convert cat to num, etc.              *
# ******************************************************************
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer  # create pipeline for ohc or standardscaling, if want to use in form of pipeline
from sklearn.impute import SimpleImputer # for missing data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import LabelEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')
    labelencoder_obj_file_path = os.path.join('artifacts', 'labelencoder.pkl')

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This fuction is resposible for Data Transformation
        
        '''

        try:
            numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
            categorical_columns =['Gender',
                    'family_history_with_overweight',
                    'FAVC',
                    'CAEC',
                    'SMOKE',
                    'SCC',
                    'CALC',
                    'MTRANS']

            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")), # replacing with mode
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical Columns: {categorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            labelencoding = LabelEncoder()

            target_column_name = "NObeyesdad"
            numerical_columns = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Applying preprocesing object on training and testing dataframe")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            target_feature_train_arr = labelencoding.fit_transform(target_feature_train_df )
            target_feature_test_arr = labelencoding.transform(target_feature_test_df )



            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_arr)
            ]

            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_arr)
            ]

            logging.info(f"Saved preprocessing object")

            save_object(
                file_path =  self.data_transformation_config.preprocessor_obj_file_path,obj=preprocessing_obj
            )
            save_object(
                file_path =  self.data_transformation_config.labelencoder_obj_file_path,obj=labelencoding
            )

            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)
        
