from fileinput import filename
import cv2
import numpy as np

#Init camera

cap = cv2.VideoCapture(0)

#Face Detection using haarcascade File

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_alt.xml")

skip = 0

face_data = []
dataset_path = './data/'
filename=input("enter the name:")

while True:
    ret,frame = cap.read()
    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(frame,1.3,5)

    #The next line of code is written to only store the largest face in the window frame
    faces = sorted(faces,key = lambda  f: f[2]*f[3])

    #start sorting from the last face since the last face is the largest in terms of area(w*h)
    for face in faces[-1:] :
        x,y,w,h = face
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

        #extract the required face or the region of the interest
        #Refers to adding an extra 10 pixels on all the sides of the required extracted face
        offset = 10
        #By default face slicing is done in (y,x) manner
        face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section = cv2.resize(face_section,(100,100))
        skip+=1
        if skip%10==0 : #Store every 10th frame
            face_data.append(face_section)
            print(len(face_data)) #number of faces captured so far

    cv2.imshow("Video Frame",frame)
    cv2.imshow("Face section frame",face_section)
    key_pressed = cv2.waitKey(1) & 0xFF
    if key_pressed == ord('q'):
        break
face_data=np.asarray(face_data)
face_data=face_data.reshape(face_data.shape[0],-1)
print(face_data.shape)

np.save(dataset_path+filename+'.npy',face_data)
print("Data saved successfully at"+dataset_path+filename+'.npy')

cap.release()
cv2.destroyAllWindows()