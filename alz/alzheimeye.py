
import cv2

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    classNames = []
   
    classFile = 'C:\\Users\\Renga Bashyam\\Desktop\\alz\\coco.txt'
    
    #This is just used to copy the lists into classnames
    with open(classFile,'rt')as f:
        classNames=f.read().rstrip('\n').split('\n')
    
    configPath = 'C:\\Users\\Renga Bashyam\\Desktop\\alz\\yolo3.cfg'
    
    weightspath ='C:\\Users\\Renga Bashyam\\Desktop\\alz\\yolo3.weights'
    
    net = cv2.dnn_DetectionModel(weightspath,configPath)
    #passing image with weights and params
    
    net.setInputSize(320,320)
    net.setInputScale(1.0/127.5)
    net.setInputMean((127.5,127.5,127.5))
    net.setInputSwapRB(True)
    # docs default parms
    if cap.isOpened():
        ret,frame=cap.read()
    else:
        ret = False
        
    #just checking if the cam is opened and any frame is registered
    while True:
        ret,frame=cap.read()
    #while loop to recur the vid to frames
        
    classIds,confs,bbox = net.detect(frame,confThreshold=0.5)
    #params to detect with threshold more than 50% kinda used to measure the accuracy
    
    print(classIds,bbox)
    #getting info from all three params and storing it in three variables
    for classIds, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
        
        #just to create a green rectangle box
        cv2.rectangle  (frame,box,color=(0,255,0),thickness=2)
        
        cv2.putText(frame,classNames[classIds-1].upper,(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #just to write the text mentioning the object here
        cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30).cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #just to write the accuracy of the img seen
        cv2.imshow('Output vid',frame)
        
        if cv2.waitKey(1)==27:
            break
        # to break the function by pressing esc key 
    cv2.destroyAllWindows()
    cap.release()
    
if __name__=="__main__":
    main()
        





