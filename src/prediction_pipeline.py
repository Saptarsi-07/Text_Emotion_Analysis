import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging

model_path='model_classifier.pickle'

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,X):
        try:
           
            with open(model_path,'rb') as file_obj:
               model=pickle.load(file_obj)

        
            pred=model.predict(X)
            return pred
        
        except Exception as e:
            logging.info('Exception at transformation stage prediction pipeline')
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 text:str):
                
        
        self.text=text
        

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'text':[self.text],
                
            }

            df=pd.DataFrame(custom_data_input_dict)
            logging.info('Converted to Dataframe')
            return df
        
        except Exception as e:
            logging.info('Exception at prediction pipeline')
            raise CustomException(e,sys)