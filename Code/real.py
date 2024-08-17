import cv2
import numpy as np
from ultralytics import YOLO 
import matplotlib.pyplot as plt 
from PIL import Image
import os
import MyUtils
import Paths
Path=Paths.ProjectPath()
model=YOLO(os.path.join(Path.Yolo_Training_Logs_Path,"resultx-2//weights//best.pt"))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    
    
    res=model.predict(frame)
    try:

        masks=res[0].masks.data
        boxes=res[0].boxes.xywhn.numpy()
        scores=res[0].boxes.conf
        n=masks.shape[0]

        r,g,b=cv2.split(frame)
        
        for m in range(n):
            
            #mask=cv2.resize(masks[m,:,:].numpy().astype(np.uint8),(width,height))
            mask=masks[m,:,:].numpy().astype(np.uint8)
            for i in range(height):
                for j in range(width):
                    if mask[i][j]>0:
                            b[i][j]=255
                        
        
        frame=cv2.merge([r,g,b])

        for i in range(n):
            box=list(boxes[i])
            confidence=scores[i]
            x,y,w,h=box
            # print("xywh:",x,y,w,h)
            x = x* width
            y= y * height
            bwidth = w* width
            bheight = h * height

            x = int(x - (bwidth / 2))
            y = int(y - (bheight/ 2))
            
            w = int(bwidth)
            h = int(bheight)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,0), 4)
            cv2.putText(frame, str(confidence), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #frame=MyUtils.display_instances_real_time(frame,MyUtils.convertYOLO_bb_2_MRCNN_bb(boxes,height,width),masks,np.ones(len(boxes),dtype=np.int32),['Background','Grapes'],scores)
        # print("done")
    except Exception as e:
        print(e)
 
    cv2.imshow('ViniScope', frame)
   
 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
