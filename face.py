import cv2
import numpy as np
import os
from PIL import Image
import face
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
                print(id)
                
                if (id == 9320):
                    Motor()
                    return()
                    
                    
                    
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
    print("hello")
        
def unknown():
    an = input("새로운 사용자입니다. 등록하시겠습니까? : ")
    if (an == "Yes"):
        capture()
            
    if (an == "No"):
        exit()
    
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
        

    
      
         
