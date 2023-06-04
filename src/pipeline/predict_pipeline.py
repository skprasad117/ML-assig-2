import os
import sys
import pandas as pd

from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            print("here")
            model_path = os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            label_encoder_path = os.path.join('artifacts','labelencoder.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            labelencoder=load_object(file_path=label_encoder_path)
            print("After Loading")
            print(features)
            data_scaled=preprocessor.transform(features)
            print("data transformed")
            preds=model.predict(data_scaled)
            preds = labelencoder.inverse_transform(preds.astype(int))
            return preds
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self,gender:str,age: int,height: int,weight: int,family_history_with_overweight: str,favc: str,fcvc: int,
                    ncp: int,caec: str,smoke: str,CH2O: int,scc: str,faf: int,tue: int,calc: str,mtrans:int):
        self.gender = gender
        self.age = age
        self.height = height
        self.weight = weight
        self.family_history_with_overweight = family_history_with_overweight
        self.favc = favc
        self.fcvc = fcvc
        self.ncp = ncp
        self.caec = caec
        self.smoke = smoke
        self.CH2O = CH2O
        self.scc = scc
        self.faf = faf
        self.tue = tue
        self.calc = calc
        self.mtrans = mtrans


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender": [self.gender],
                "Age": [self.age],
                "Height": [self.height],
                "Weight": [self.weight],
                "family_history_with_overweight": [self.family_history_with_overweight],
                "FAVC": [self.favc],
                "FCVC": [self.fcvc],
                "NCP": [self.ncp],
                "CAEC": [self.caec],
                "SMOKE": [self.smoke],
                "CH2O": [self.CH2O],
                "SCC": [self.scc],
                "FAF": [self.faf],
                "TUE": [self.tue],
                "CALC": [self.calc],
                "MTRANS": [self.mtrans],


            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

