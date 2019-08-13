import os

import cv2

import numpy as np
ms=0
vk=0
a=0
m=[]
n=[]


def faceDetection(test_img):

    gray_img=cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale

    face_haar_cascade=cv2.CascadeClassifier(r'C:\Users\SAMARTH G VASIST\Anaconda3\pkgs\opencv-3.1.0-np111py36_1\Library\etc\haarcascades\haarcascade_frontalface_default.xml')#Load haar classifier

    faces=face_haar_cascade.detectMultiScale(gray_img,scaleFactor=1.32,minNeighbors=5)#detectMultiScale returns rectangles



    return faces,gray_img



#Given a directory below function returns part of gray_img which is face alongwith its label/ID

def labels_for_training_data(directory):

    faces=[]

    faceID=[]



    for path,subdirnames,filenames in os.walk(directory):

        for filename in filenames:

            if filename.startswith("."):

                print("Skipping system file")#Skipping files that startwith .

                continue



            id=os.path.basename(path)#fetching subdirectory names

            img_path=os.path.join(path,filename)#fetching image path

            print("img_path:",img_path)

            print("id:",id)

            test_img=cv2.imread(img_path)#loading each image one by one

            if test_img is None:

                print("Image not loaded properly")

                continue

            faces_rect,gray_img=faceDetection(test_img)#Calling faceDetection function to return faces detected in particular image

            if len(faces_rect)!=1:

               continue #Since we are assuming only single person images are being fed to classifier

            (x,y,w,h)=faces_rect[0]

            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image

            faces.append(roi_gray)

            faceID.append(int(id))

    return faces,faceID





#Below function trains haar classifier and takes faces,faceID returned by previous function as its arguments

def train_classifier(faces,faceID):

    face_recognizer=cv2.face.LBPHFaceRecognizer_create()

    face_recognizer.train(faces,np.array(faceID))

    return face_recognizer



#Below function draws bounding boxes around detected face in image

def draw_rect(test_img,face):

    (x,y,w,h)=face

    cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=5)



#Below function writes name of person for detected label

def put_text(test_img,text,x,y):

    cv2.putText(test_img,text,(x,y),cv2.FONT_HERSHEY_DUPLEX,2,(255,0,0),4)






test_img=cv2.imread(r"C:\Users\SAMARTH G VASIST\trainingimages\1\GettyImages-689381236-e1517238885876.jpg")#test_img path

faces_detected,gray_img=faceDetection(test_img)

print("faces_detected:",faces_detected)




#This module captures images via webcam and performs face recognition

faces,faceID= labels_for_training_data(r'C:\Users\SAMARTH G VASIST\trainingimages')

face_recognizer= train_classifier(faces,faceID)





name = {0 : "Dhoni",1 : "Kohli"}




cap=cv2.VideoCapture(0)



while True:
    a+=1
    print(a)

    ret,test_img=cap.read()# captures frame and returns boolean value and captured image

    faces_detected,gray_img= faceDetection(test_img)







    for (x,y,w,h) in faces_detected:

      cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)



    resized_img = cv2.resize(test_img, (1000, 700))

    cv2.imshow('face detection Tutorial ',resized_img)

    cv2.waitKey(10)





    for face in faces_detected:

        (x,y,w,h)=face

        roi_gray=gray_img[y:y+w, x:x+h]

        label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image

        print("confidence:",confidence)

        print("label:",label)
       
        if label==0:
            ms+=1
        elif label==1:
            vk+=1
        
        

        draw_rect(test_img,face)

        predicted_name=name[label]
        print(predicted_name)
        font=cv2.FONT_HERSHEY_SIMPLEX
        color=(255,255,255)
        stroke=2
        cv2.putText(test_img,predicted_name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        #if confidence < 39:#If confidence less than 37 then don't print predicted face text on screen

        
        




    resized_img = cv2.resize(test_img, (1000, 700))
    m.append(ms)
    n.append(vk)
    
    cv2.imshow('face recognition tutorial ',resized_img)
    if a==150:
        if sum(m)>sum(n):
           print("Dhoni for sure")
          
        elif sum(n)>sum(m):
           print("Kohli for sure") 
            
        break 
               
        
        
cap.release()

cv2.destroyAllWindows
