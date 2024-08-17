from flask import Flask, jsonify, request,send_file
from flask_restful import Resource, Api,reqparse
from ultralytics import YOLO 
import pickle
import pandas as pd
import numpy as np
from flask_cors import CORS
import os
import cv2
import Paths
from PIL import Image
import io

app = Flask(__name__)
api=Api(app)

CORS(app)

Path=Paths.ProjectPath()

model=YOLO(os.path.join(Path.Yolo_Training_Logs_Path,"resultx-2//weights//best.pt"))
height=640
width=640

    
class prediction(Resource):
    
    def post(self):
        images=[]
        i=0
        image_file = request.files['img'+str(0)]
        image_path = 'img'+str(0)+'.jpg'
        image_file.save(os.path.join("E://vsc2.0//GitHub//Grape_instance_segementation_and_masking//WebPortal",image_path))

        img=cv2.imread(os.path.join("E://vsc2.0//GitHub//Grape_instance_segementation_and_masking//WebPortal",image_path))
                
        img = cv2.resize(img,(640,640),interpolation=cv2.INTER_CUBIC)
        res=model.predict(img)

        try:
        
            masks=res[0].masks.data
            boxes=res[0].boxes.xywhn.numpy()
            scores=res[0].boxes.conf
            n=masks.shape[0]
            r,g,b=cv2.split(img)
        
            for m in range(n):
                mask=masks[m,:,:].numpy().astype(np.uint8)
                for i in range(height):
                    for j in range(width):
                        if mask[i][j]>0:
                                r[i][j]=255
        
            img=cv2.merge([r,g,b])

            for i in range(n):
                box=list(boxes[i])
                confidence=scores[i]
                x,y,w,h=box
                print("xywh:",x,y,w,h)
                x = x* width
                y= y * height
                bwidth = w* width
                bheight = h * height

                
                x = int(x - (bwidth / 2))
                y = int(y - (bheight/ 2))
                w = int(bwidth)
                h = int(bheight)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0,0), 1)
                cv2.putText(img, str(confidence), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            img_byte_arr = io.BytesIO()
            img_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)
            image_path = 'img'+str(0)+'.jpg'
            os.remove(os.path.join("E://vsc2.0//GitHub//Grape_instance_segementation_and_masking//WebPortal",image_path))
            return send_file(img_byte_arr, mimetype='image/jpeg')
        
        except Exception as e:
            return {"error": str(e)}, 500
            print(e)
        
            


api.add_resource(prediction, '/predict')
  

if __name__ == '__main__':
  
    app.run(port=3000,debug = True)