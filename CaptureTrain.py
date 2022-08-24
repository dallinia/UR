#q를 누르면 얼굴을 찍을 것이냐고 물어본다. 여기서 Yes를 치면 사진을 총 20장 찍게 되고 바로 trainer폴더에 학습시킨 데이터를 xml파일로 저장한다.
#NO를 칠 경우 아무것도 이루어지지 않고 다시 q를 누르면 다시 물어보게 된다.


import cv2
import os
import keyboard
import numpy as np
from PIL import Image
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()



def capture():
    count2 = 0
    while True:
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # For each person, enter one numeric face id
        face_id = input("enter your id: ")
        print("\n [INFO] Initializing face capture. Look the camera and wait ...")
        # Initialize individual sampling face count
                
        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
            faceSamples=[]
            ids = []
            for imagePath in imagePaths:
                PIL_img = Image.open(imagePath).convert('L') # convert it to grayscale
                img_numpy = np.array(PIL_img,'uint8')
                id = int(os.path.split(imagePath)[-1].split(".")[1])
                faces = face_detector.detectMultiScale(img_numpy)
                for (x,y,w,h) in faces:
                    faceSamples.append(img_numpy[y:y+h,x:x+w])
                    ids.append(id)
            return faceSamples,ids


        
        
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
                count2 += 1
                # Save the captured image into the datasets folder
                cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count2) + ".jpg", gray[y:y+h,x:x+w])
                #cv2.imshow('image', img)
            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
            #if k == 27:
                #break
            if count2 >= 20: # Take 30 face sample and stop video
                count2 = 0
                break
                    
        print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
        faces,ids = getImagesAndLabels(path)
        recognizer.train(faces, np.array(ids))
        recognizer.write('trainer/trainer.yml')
        print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
        break
# Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
