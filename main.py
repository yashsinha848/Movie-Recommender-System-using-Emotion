from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
def emo():
    face_classifier = cv2.CascadeClassifier(r'C:\Machine Learning\Emotion Recognition\haarcascade_frontalface_default.xml')
    classifier = load_model(r'C:\Machine Learning\Emotion Recognition\Emotion_little_vgg_updated.h5')

    class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

    cap = cv2.VideoCapture(0)

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            # rect,face,image = face_detector(frame)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # make a prediction on the ROI, then lookup the class

                preds = classifier.predict(roi)[0]
                label = class_labels[preds.argmax()]
                label_position = (x, y)
                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (14, 219, 231), 3)
                return label
            else:
                cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (14, 219, 231), 3)
        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
emr=emo()
import pandas as pd
emotion=emr
print('Emotion is:',emotion)
if(emotion=="Angry"):
    df=pd.read_csv(r"C:\4 Semester\Project Exhibition-II\angry-movies-genre.csv")
    #print(df.head())
if(emotion=="Surprise"):
    df=pd.read_csv(r"C:\4 Semester\Project Exhibition-II\surprise-movies-genre.csv")
    #print(df.head())
if(emotion=="Happy"):
    df=pd.read_csv(r"C:\4 Semester\Project Exhibition-II\happy-movies-genre.csv")
    #print(df.head())
if(emotion=="Neutral"):
    df=pd.read_csv(r"C:\4 Semester\Project Exhibition-II\neutral-movies-genre.csv")
    #print(df.head())
if(emotion=="Sad"):
    df=pd.read_csv(r"C:\4 Semester\Project Exhibition-II\sad-movies-genre.csv")
    #print(df.head())
genres_list=['Adventure','Animation','Children','Comedy','Fantasy','Romance','Horror','Sci-Fi','Western','Action','Drama','Thriller','IMAX','Crime','War']
top=[]
for i in genres_list:
    df3=df[df['genres'].str.contains(i)]
    top.append(df3)
genres_dict={'Angry':['Horror','Action','Thriller','IMAX'],'Happy':['Adventure','Animation','Children','Comedy','Fantasy',],'Surprise':['Adventure','Fanatsy','Sci-Fi','Horror'],'Neutral':['Adventure','Comedy','Romance','Drama','Western','Thriller'],'Sad':['Comedy','Romance','Action']}
a=dict()
for i in range(len(genres_list)):
    a[genres_list[i]]=top[i]
for j in range(len(genres_dict[emotion])):
    print('For ',genres_dict[emotion][j],' Genre Top movies are-')
    print(a[genres_dict[emotion][j]].head())

movie_name=input("Enter the name of the movie which you have watched previously and want to watch similar movie to that:")
from MovieRecommends import *
movie_recommender(movie_name)






























