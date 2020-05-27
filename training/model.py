import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense,Conv2D,MaxPooling2D,Dropout,BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import matplotlib.pyplot as plt

class Model:
    """
    This Class Train the model and save the best model according to validation
    aaccuracy in the form of .h5 file

    Written By : Bhisma
    
    """
    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def cnn_model(self):
        """
        Method Name: cnn_model
        Description : This method defines the Conventional Neural Network Model
                    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
                    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
                    [2 x CONV (3x3)] — MAXP (2x2) — DROPOUT (0.5)
                    Dense (256) — DROPOUT (0.4)
                    Dense (128) — DROPOUT (0.5)

        """
        try:
            num_features = 64
            num_labels = 7
            width, height = 48, 48
            self.train_aug = ImageDataGenerator(rescale=1./255,rotation_range=20,horizontal_flip=True,zoom_range=0.2,shear_range=0.2)
            self.test_aug = ImageDataGenerator(rescale=1./255)

            model = Sequential()

            model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last'))
            model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
            model.add(BatchNormalization())
            model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            model.add(Dropout(0.5))

            model.add(Flatten())

            model.add(Dense(2*2*num_features, activation='relu'))
            model.add(Dropout(0.4))
        
            model.add(Dense(2*num_features, activation='relu'))
            model.add(Dropout(0.5))

            model.add(Dense(num_labels, activation='softmax'))

            return model

        except Exception as e:
            print("Exceptions occurs",str(e))

            
            
            
    def training(self,model):
        """
        Method Name: training
        Description : Now the model that is pass from above method started to train
                      and the accuracy and loss of training process is saved in
                      "training_images/" folders.I have use "adam" as an optimizer
                      "categorical_crossentropy" as a loss .

        """
        try:
            BATCH_SIZE = 64
            EPOCHS = 100       
            
            model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

            lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1)

            early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1,
                                        mode='auto')

            checkpoint = ModelCheckpoint('training/emotion_model.h5',monitor='val_loss',verbose=0,
                                        save_best_only='True',mode='auto')

            history = model.fit(self.train_aug.flow(self.X_train,self.y_train,batch_size=BATCH_SIZE),
                               callbacks=[lr_reducer,early_stopper,checkpoint],epochs=EPOCHS,
                               validation_data=self.test_aug.flow(self.X_test,self.y_test))
            
            #plotting training and Validation (accuraccy and loss)
            plt.plot(history.history["accuracy"])
            plt.title("Model Accuracy")
            plt.ylabel("Accuracy")
            plt.xlabel("Epochs")
            plt.legend(["Train","Validation"],loc = 'upper_left')
            plt.savefig("training_images/accuracy.png")

            plt.plot(history.history["loss"])
            plt.title("Model Loss")
            plt.ylabel("Loss")
            plt.xlabel("Epochs")
            plt.legend(["Train","Validation"],loc = 'upper_left')
            plt.savefig("training_images/loss.png")
            

        except Exception as e:
            print("Exception occurs",str(e))

