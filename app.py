import cv2
import numpy as np
from tensorflow.keras.models import load_model

faces = cv2.CascadeClassifier("fer2013/haarcascade_frontalface_default.xml")

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

model = load_model("training/emotion_model.h5")

video = cv2.VideoCapture(0)

while True:
    _,frame = video.read()
    image_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces_region = faces.detectMultiScale(image_gray,1.3,3)
    for (x,y,w,h) in faces_region:
        roi = image_gray[x:x+w,y:y+h]
        image_resize = np.resize(roi,(48,48))
        resized = image_resize/255
        image = np.reshape(resized,(1,48,48,1))
        result = model.predict(image)
        label = np.argmax(result,axis=1)[0]
        emotion = emotions[label]
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)
        cv2.putText(frame, emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv2.resize(frame, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

video.release()
cv2.destroyAllWindows()
