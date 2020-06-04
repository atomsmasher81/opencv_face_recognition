import os
from collections import Counter

import numpy as np
import cv2.cv2 as cv2

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
from rest_framework.response import Response

from face_recognizer.recogniser import prepare_training_data, FaceRecognizer, mypredict
from security_cam_web_app.settings import BASE_DIR

# faceCascade = cv2.CascadeClassifier( '/mnt/54C615C6C615A8EE/my_projects/security_cam_web_app/face_recognizer/opencv-files/haarcascade_frontalface_alt.xml')
faceCascade = cv2.CascadeClassifier( f'{BASE_DIR}/face_recognizer/opencv-files/haarcascade_frontalface_alt.xml')

def go_live(user_id):
    try:
        cap = cv2.VideoCapture(0)
        cap.set(3, 640)  # set Width
        cap.set(4, 480)  # set Height
        os.makedirs(BASE_DIR + f'/media/preprocessed/{user_id}/', exist_ok=True)
        os.makedirs(BASE_DIR + f'/media/postprocessed/{user_id}/', exist_ok=True)
        start_count = count = len(os.listdir(BASE_DIR + f'/media/preprocessed/{user_id}/'))
        print('~~~~~~~~~~~',count)
        while True:
            ret, img = cap.read()
            # img = cv2.flip(img, -1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,

                scaleFactor=1.2,
                minNeighbors=5
                ,
                minSize=(20, 20)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                cv2.imwrite(BASE_DIR + f'/media/preprocessed/{user_id}/{count}.jpg', img)
                count +=1
            cv2.imshow('video', img)

            k = cv2.waitKey(30) & 0xff
            if k == 27:  # press 'ESC' to quit

                break
    except:
        print('&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
    finally:
        cap.release()
    subjects = ['kartik',]
    pics = os.listdir(BASE_DIR + f'/media/preprocessed/{user_id}/') [start_count:]
    faces,labels = prepare_training_data(BASE_DIR + f'/media/trusted_people/{user_id}/')
    print("@@@@@@@@@@@@@@@@@@@@@@2   data prepared ")
    model = FaceRecognizer()
    model.train(faces,labels)
    print("@@@@@@@@@@@@@@@@@@@@@@2   model trained ")
    print(pics)
    name_count = Counter(map(lambda x: x.split('-')[0],os.listdir(BASE_DIR + f'/media/postprocessed/{user_id}/')))

    for pic in pics:
        try:
            actual_pic = cv2.imread(BASE_DIR + f'/media/preprocessed/{user_id}/{pic}')
            img,name = mypredict(actual_pic, subjects)
            cv2.imwrite(BASE_DIR + f'/media/postprocessed/{user_id}/{name}_{name_count[name] + 1}.jpg',img)
        except Exception as e:
            print("i am here , ",e)

    return Response({'message':'success'})

# go_live(1)