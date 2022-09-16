import cv2
import numpy as np
import os
from PIL import Image
import random

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
Scaffolding=dict()
sf_all={1,2,3,4,5,6,7,8,9}
sf_owner=set(list(Scaffolding))
sf_ownerless = sf_all 
path = 'dataset'
id_path = "ID.txt"
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'
dir_path = 'trainer/trainer.yml'
recognizer.read('trainer/trainer.yml')

class Captureface:
    
    def capture():
        global face_id
        
        count2 = 0
        while True:
            cam = cv2.VideoCapture(0)
            cam.set(3, 640) # set video width
            cam.set(4, 480) # set video height
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            # For each person, enter one numeric face id
            f = open('./ID.txt', 'r')
            data = f.read()
            data_into_list = data.split("\n")
            f.close()
            face_id = str(input('enter our id: '))
            if (face_id in data_into_list):
                print("이미 존재하는 아이디 입니다.")
                return()
            data_into_list.append(face_id)
            print("\n [INFO] Initializing face capture. Look the camera and wait ...")
            with open('ID.txt','w', encoding='UTF-8') as f:
                for face_id in data_into_list:
                    f.write(face_id + '\n')
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
                
            add_id()           
            print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
            faces,ids = getImagesAndLabels(path)
            recognizer.train(faces, np.array(ids))
            recognizer.write('trainer/trainer.yml')
            print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
            break
    # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        DelecteAllFile(path)
        exit()
        
    def show_cam(previewName, cam_num):
    
        f = open('./ID.txt', 'r')
        data = f.read()
        data_into_list = data.split("\n")
        f.close()
        
        cam = cv2.VideoCapture(cam_num)
        minW = 0.1*cam.get(3)
        minH = 0.1*cam.get(4)
    

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
                            Motor()
                        
                        elif (AN == 'No'):
                            break
                    
                    else:
                        print("등록되지 않은 아이디 입니다.")
                        unknown()
                        
                        
                else:
                    id = "unknown"
                    confidence = "  {0}%".format(round(100 - confidence))
                    unknown()
            
            cv2.imshow('camera' ,img) 
            cv2.imshow('camera' ,img)
            k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
            if k == 27:#esc
                break
                    
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        
        
        
def DelecteAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'RemoveAllFile'
    else:
        return 'Directory Not Found'
        
        
def delsf():
    global Scaffolding
    global sf_owner
    global sf_ownerless
    global face_id
    global sf_all
    face_id = int(input('4자리 숫자 (삭제번호)입력하시오. : '))
    if face_id >0 : 
        mir_sf={v:k for k,v in Scaffolding.items()} #ket:value 반전시켜서 삭제 
        if face_id in mir_sf:
            del mir_sf[face_id]
            Scaffolding={v:k for k,v in mir_sf.items()}
            sf_owner=set(list(Scaffolding))
            sf_ownerless=sf_all - sf_owner
            print(f"삭제 완료 Scaffolding{Scaffolding}, 주인ㅇ {sf_owner}, 주인ㄴ{sf_ownerless}")
                        
    elif face_id == 0:
        print("삭제과정을 생략합니다.")  
            
def add_id() :
    global sf_ownerless 
    global face_id
    global Scaffolding
                
                
    if face_id > str(0) :
        sf_ownerless = list(sf_ownerless)
        owner=random.choice(sf_ownerless)
        Scaffolding[owner]=face_id #key:value이렇게 맞춰야 실행됨(1:3829 <-이런형태)
                
    sf_ownerless = set(sf_ownerless)    
                
    for i in sf_all:
        if i in Scaffolding:
            sf_owner.add(i)
                        
            sf_ownerless = sf_all - sf_owner
                
    print(f"추가 완료 Scaffolding{Scaffolding}, 주인ㅇ {sf_owner}, 주인ㄴ{sf_ownerless}")
        
def Motor():
    global id
    global Scaffolding
        
    print("hello")
    exit()
        
        
    
def unknown():
    an = input("새로운 사용자입니다. 등록하시겠습니까? : ")
    if (an == "Yes"):
        Captureface.capture()
                
    if (an == "No"):
        exit()
                        
def Delect_persion():
    global data_into_list
    global id
        
    f = open('./ID.txt', 'r')
    data = f.read()
    data_into_list = data.split("\n")
    f.close()
        
    cam = cv2.VideoCapture(0)
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)
    

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
                    ask()
        
                else:
                    print("등록되지 않은 아이디 입니다.")
                    exit()
                        
def ask():
    global data_into_list
    global id
        

    ask = input("아이디를 삭제하시겠습니까?: ")
    if (ask == "Yes"):
        Captureface.delsf()
        while id in data_into_list:
            data_into_list.remove(id)


        with open('ID.txt','w', encoding='UTF-8') as f:
            for id in data_into_list:
                f.write(id + '\n')
                exit()
                        
    if (ask == "No"):
        exit()
        
        

if __name__ == '__main__':
    name = input()
    if (name == '1'):
        Captureface.show_cam("front", 0)
        
    if (name == '2'):
        Captureface.capture()
        
        
    if (name == '3'):
        Delect_persion()
                

