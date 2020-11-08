import cv2
import datetime
import winsound
import smtplib
import requests
import bs4
import time


def main():
    
    now = datetime.datetime.now().time()
    if now.hour == 11 and now.minute == 0:
        success = True
        freq = 100
        dur = 50
          
         
        for i in range(0, 10):     
            winsound.Beep(freq, dur)     
            freq+= 100
            dur+= 50
            print("It's 11:00 time for medicine!")
        
    if now.hour == 10 and now.minute == 8:
        success = True
        freq = 100
        dur = 50
          
         
        for i in range(0, 10):     
            winsound.Beep(freq, dur)     
            freq+= 100
            dur+= 50
            print("It's 1:00 time for lunch!")    
    
    
    thres = 0.5
    
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    
    
    
    
    
    classNames=[]
    classFile='C:\\Users\\Renga Bashyam\\Desktop\\alz\\coco.names'
    with open(classFile,'rt') as f:
        classNames=f.read().rstrip('\n').split('\n')
    configPath = 'C:\\Users\\Renga Bashyam\\Desktop\\alz\\yolo3.cfg'
    weightsPath = 'C:\\Users\\Renga Bashyam\\Desktop\\alz\\yolov3.weights'
    
    net = cv2.dnn_DetectionModel(weightsPath,configPath)
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean(127.5)
    net.setInputSwapRB(True)
    
    while True:
        success,img=cap.read()
        classIds , confs,bbox = net.detect(img,confThreshold=thres)
        print(classIds,bbox)
        
        if len(classIds) != 0:
        
            for classId,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            
            
        
        
        
        cv2.imshow('Output',img)
        cv2.waitKey(1)
        
        if classNames[classId].upper() != 'PERSON':
            
            server=smtplib.SMTP('smtp.gmail.com',587)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login('1justb3@gmail.com','Thisisatest@1')     #Email and App Passsword to be entered.#
            subject='Patient not found in front of the camera pls call him immedietly'
            
            msg=f"Subject:{subject}\n\n"
            server.sendmail(
                '1justb3@gmail.com',          #Email of Sender#
                'mibashyam@gmail.com',          #Email of Reciever#
                msg
            )
            print('Hey Email has been sent!')
            server.quit()
        
if __name__=='__main__':
    main()
