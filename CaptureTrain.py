import cv2
import os
import numpy as np
from PIL import Image


path = 'dataset'
id_path = "ID.txt"
recognizer = cv2.face.LBPHFaceRecognizer_create()

#if not os.path.isfile(path):
    #f = open("sp500.txt", 'w')

def capture():
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
        face_id = input('enter our id: ')
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
    
    
def DelecteAllFile(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'RemoveAllFile'
    else:
        return 'Directory Not Found'
    
def Delect_persion():
    f = open('./ID.txt', 'r')
    data = f.read()
    data_into_list = data.split("\n")
    f.close()

    ID = input("삭제할 아이디를 입력해주세요: ")
    while ID in data_into_list:
        data_into_list.remove(ID)


    with open('ID.txt','w', encoding='UTF-8') as f:
        for ID in data_into_list:
            f.write(ID + '\n')
            