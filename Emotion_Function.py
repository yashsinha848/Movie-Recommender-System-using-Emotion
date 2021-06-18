
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(r'C:\Machine Learning\Emotion Recognition\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Machine Learning\Emotion Recognition\Emotion_little_vgg_updated.h5')

class_labels = ['Angry','Happy','Neutral','Sad','Surprise']


def emotion():
    cap = cv2.VideoCapture(0)




# Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
        # rect,face,image = face_detector(frame)
    
    
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)
    
            # make a prediction on the ROI, then lookup the class
    
            preds = classifier.predict(roi)[0]
            label=class_labels[preds.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(14, 219, 231),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(14, 219, 231),3)
    cv2.imshow('Emotion Recognition',frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
     #   break
    return label
    cap.release()
    cv2.destroyAllWindows()

    


a=emotion()
print(a)






















