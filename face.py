import cv2
import numpy as np
import os
from PIL import Image
import CaptureTrain


path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
i = 0

dir_path = 'trainer/trainer.yml'
if (os.path.isfile(dir_path) == False):
    CaptureTrain.capture()
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX


k = cv2.waitKey(10) & 0xff

def show_cam(previewName, cam_num):
    
    f = open('./ID.txt', 'r')
    data = f.read()
    data_into_list = data.split("\n")
    f.close()
    
    cam = cv2.VideoCapture(cam_num)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    video = -1 # 객체를 담을 수 있는 변수 선언(-1로 초기화한거임.)
   

    while True:
        ret, img =cam.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale( 
            gray,
            scaleFactor = 1.2,
            minNeighbors = 5,
            minSize = (int(minW), int(minH)),
        )
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
            # Check if confidence is less them 100 ==> "0" is perfect match 
            if (confidence < 80):
                id = id
                confidence = "  {0}%".format(round(100 - confidence))
                
                
                if (str(id) in data_into_list):
                    print(id)
                    AN = input("당신의 아이디가 맞습니까?: ")
                    if (AN == 'Yes'):
                        print("Hello" + ' ' + str(id))
                        return()
                    
                    elif (AN == 'No'):
                        break
                
                else:
                    print("등록되지 않은 아이디 입니다.")
                    unknown()
                    
                    
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
                unknown()
                
            
            cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        cv2.imshow('camera' ,img) 
        cv2.imshow('camera' ,img)
        k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
        if k == 27:#esc
            break
                  
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    
def Motor():
    global id
    
    print("hello" + str(id))
        
def unknown():
    an = input("새로운 사용자입니다. 등록하시겠습니까? : ")
    if (an == "Yes"):
        CaptureTrain.capture()
            
    if (an == "No"):
        exit()
    
