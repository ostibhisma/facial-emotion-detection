import numpy as np
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical

class DataLoader:
    """
    This class shall obtain a data and preprocess the data for training
    Written By: Bhisma

    """
    def __init__(self,path_to_data):
        self.path_to_data = path_to_data
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []


    def get_data(self):
        """
        Method Name : get_data
        Description : This method read the data using pandas dataframe and according
                      to the Training or Testion given in the 'Usage' column in df
                      it load the training and testing data and convert it to the 
                      numpy array (Because image is commonly stored as numpy array for 
                      training the model)  and  reshape the data in (48,48,1) format and
                      returns the training and testing data
        Written By: Bhisma 

        """
        try:
            df = pd.read_csv(self.path_to_data)
            for index,rows in tqdm(df.iterrows()):
                x = rows['pixels'].split(" ")
                if rows['Usage'] =='Training':
                    self.X_train.append(np.array(x,'float32'))
                    self.y_train.append(rows['emotion'])
                else:
                    self.X_test.append(np.array(x,"float32"))
                    self.y_test.append(rows['emotion'])
            self.X_train = np.array(self.X_train)
            self.X_test = np.array(self.X_test)
            self.y_train = np.array(self.y_train)
            self.y_test = np.array(self.y_test)
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)
            self.X_train = self.X_train.reshape(self.X_train.shape[0],48,48,1)
            self.X_test = self.X_test.reshape(self.X_test.shape[0],48,48,1)
            return self.X_train,self.X_test,self.y_train,self.y_test
        
        except Exception as e:
            print("Exception occurs",str(e))