from data_loading.data import DataLoader
from training.model import Model

def training_model():
    """
    function Name : training_model
    Description : This is a main function in which it loads all the data
                  and train the model and save the best model in .h5 format
                  in "training/emotion_model.h5"
    Written By : Bhisma
    """
    dl = DataLoader('./fer2013/fer2013.csv')
    X_train,X_test,y_train,y_test = dl.get_data()
    model_object = Model(X_train,X_test,y_train,y_test)
    model = model_object.cnn_model()
    model_object.training(model)


if __name__ == '__main__':
    training_model()
