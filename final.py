import rasterio
import numpy as np
import Pillow
import ultralytics
from ultralytics import YOLO
import torch 
import cv2
import csv
import pandas as pd
from PIL import Image
from io import StringIO
import os
from rasterio.windows import Window
 



def tile_generator(src,tile_size,overlap):
    for row_start in range(0,src.height,tile_size-overlap):
        for col_start in range(0,src.width,tile_size-overlap):
            row_end=min(row_start+tile_size,src.height)
            col_end=min(col_start+tile_size,src.width)

            win=Window(col_start,row_start,col_end-col_start,row_end-row_start)
            data=src.read(window=win)
            print("Inside Tile Gen")
            yield data,(row_start,col_start)   

def convert_to_RGB(tile_data):
    print('Shape',tile_data.shape)
    if tile_data.shape[0]==4 or tile_data.shape[0]==3:
        tile_data=tile_data[:3,:,:]
    return np.moveaxis(tile_data,0,-1)

def segmentation(masks, src):
    combined_mask = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)

    # Combine all masks into a single binary mask
    for mask in masks:
        mask_data = mask.data[0].cpu().numpy()
        binary_mask = (mask_data > 0.5).astype(np.uint8) * 255
        resized_mask = cv2.resize(binary_mask, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_LINEAR)
        combined_mask = cv2.bitwise_or(combined_mask, resized_mask)

    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("No contours found")
        return None  

    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.int32)

    if box is not None and len(box) == 4:
        x1, y1 = np.min(box, axis=0)
        x2, y2 = np.max(box, axis=0)
        return [x1, y1, x2, y2]  

    return None  
    
def extract_detections(results,src,model,all_detections):
    for box in results[0]:
        if box.masks:
            xyxy = segmentation(box.masks, src)
        else:
            xyxy=box.boxes.xyxy.cpu().numpy().tolist()[0]

        conf=box.boxes.conf.cpu().numpy().item()
        class_id=int(box.boxes.cls[0])
        class_name=results[0].names[class_id]
        spec=box.boxes.cls.cpu().numpy().item()
        all_detections.append({
            "xyxy":xyxy,
            "conf":conf,
            "class_name":class_name,
            "spec":spec,
            "class_id":class_id
        })
    return all_detections




'''   
def segmentation(results,src,model,all_detections):
    for mask in results[0].masks:
        mask_data=mask.data[0].cpu().numpy()
        binary_mask=(mask_data>0.5).astype(np.uint8)*255
        resized_mask=cv2.resize(binary_mask,(src.shape[1],src.shape[0]),interpolation=cv2.INTER_LINEAR)
        blurred_mask=cv2.GaussianBlur(resized_mask,(5,5),0)
        contours=cv2.findContours(blurred_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        smoothed_contours=[cv2.approxPolyDP(cnt,3,True) for cnt in contours]

        if len(contours)==0:
            print("No contours found")
        
        else:
            bounding_boxes=[]
            for contour in contours:
                if len(contour)>0:
                    rect=cv2.minAreaRect(contour)
                    box=cv2.boxPoints(rect)
                    if box is not None and len(box)==4:
                        x1,y1=np.min(box,axis=0)
                        x2,y2=np.max(box,axis=0)
                        bounding_boxes.append([x1,y1,x2,y2])
                #all_detections=extract_detections(results,src,model,all_detections)
                #all_detections[0]['xyxy']=bounding_boxes

'''     
    


def calculate_iou(box1,box2):
    x1,y1,x2,y2=box1
    x3,y3,x4,y4=box2

    xi1,yi1=max(x1,x3),max(y1,y3)
    xi2,yi2=max(x2,x4),max(y2,y4)

    intersection=max(0,xi2-xi1)*max(0,yi2-yi1)
    box1_area=(x2-x1)*(y2-y1)
    box2_area=(x4-x3)*(y4-y3)

    class_det={}
    union=box2_area+box1_area-intersection

    return intersection/union if union>0 else 0

def NMS(detections,iou_threshold=0.5):
    detections=sorted(detections,key= lambda x: x['conf'],reverse=True)
    filtered_det=[]
    while detections:
        best = detections.pop(0)
        filtered_det.append(best)
        detections = [det for det in detections
            if calculate_iou(best["xyxy"], det["xyxy"]) < iou_threshold]   
    return filtered_det  


def renanme_classes(detections):
    class_name_mapping={
        '1':'Helicopter',
        '0':'Transport'
    }

    for detection in detections:
        current_class_name=detection['class_name']
        if current_class_name in class_name_mapping:
            detection['class_name']=class_name_mapping[current_class_name]
    
    return detections

def is_not_tiff(image_path):
    return image_path.lower().endswith('.jpg')

def save_to_csv(src,col_start,row_start,detections,csv_file_tiff,csv_file,image_path):
    
    for det in detections:
        x1,y1,x2,y2=map(int,det['xyxy'])
        conf=det['conf']
        class_name=det['class_name']
        spec=det['spec']

        print('Inside Inference')
        x2=x2-x1
        y2=y2-y1

        tile_num=f'{row_start}_{col_start}'

        if(is_not_tiff(image_path)):
            with open(csv_file,mode='a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([f'{x1:.2f}',f'{y1:.2f}',f'{x2:.2f}',f'{y2:.2f}',src.width,src.height,conf,class_name,spec,tile_num])

        else:
            print('Coordinate:',x1,y1,x2,y2)
            global_x1=x1+col_start
            global_y1=y1+row_start
            global_x2=x2+col_start
            global_y2=y2+row_start

            lon_tl,lat_tl=src.transform*(global_x1,global_y1)
            lon_tr,lat_tr=src.transform*(global_x2,global_y1)
            lon_bl,lat_bl=src.transform*(global_x1,global_y2)
            lon_br,lat_br=src.transform*(global_x2,global_y2)

            with open(csv_file_tiff,mode='a',newline='') as file:
                writer=csv.writer(file)
                writer.writerow([f'{lon_tl:.2f}',f'{lat_tl:.2f}',f'{lon_tr:.2f}',f'{lat_tr:.2f}',f'{lon_bl:.2f}',f'{lat_bl:.2f}',f'{lon_br:.2f}',f'{lat_br:.2f}',src.width,src.height,conf,class_name,spec,tile_num])

def annotate_images(image,detections):
    print("inside annotate")
    for det in detections:
        x1,y1,x2,y2=map(int,det['xyxy'])
        conf=det['conf']
        label = f"{det['class_name']}{conf:.2f}"
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.putText(image.label,x1,y1-10,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
    return image



def process_tiff(tile_size,overlap):

    image_path=""

    model_paths=['']
    models=[YOLO(model_path) for model_path in model_paths]

    confs=[]
    
    total_det=0

    csv_file_tiff=""
    csv_file=""
    csv_buffere=StringIO()

    with open(csv_file_tiff, mode='w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow('lon_tl','lat_tl','lon_tr','lat_tr','lon_bl','lat_bl','lon_br','lat_br','src.width','src.height','conf','class_name','spec','tile_num')
    
    with open(csv_file, mode='w',newline='') as file:
        writer=csv.writer(file)
        writer.writerow('x1','y1','x2','y2','src.width','src.height','conf','class_name','spec','tile_num')
    
    with rasterio.open(image_path) as src:
        for tile_data,(row_start,col_start) in tile_generator(src,tile_size,overlap):
            all_detections=[]
            detect=False
            tile_image=convert_to_RGB(tile_data)
            tile_image=Image.fromarray(tile_image)

            for model_idx,model in enumerate(models):
                results=model.predict(tile_image,conf=confs[model_idx])

                if len(results[0].boxes==0):
                    total_AC=0
                else:
                    total_AC=len(results[0].boxes)
                    detect=True
                
                total_det+=total_AC
                print("Total Detections",total_det)

                all_detections=extract_detections(results,model,all_detections)
            
            all_detections=NMS(all_detections,iou_threshold=0.5)
            all_detections=renanme_classes(all_detections)

            save_to_csv(src,col_start,row_start,all_detections,csv_file_tiff,csv_file,image_path)

            tile_cv_img=np.array(tile_image)
            tile_cv_img=cv2.cvtColor(tile_cv_img,cv2.COLOR_RGB2BGR)
            tile_cv_img=annotate_images(tile_cv_img,all_detections)

            if detect==True:
                output_file=os.path.join('',f'{row_start}_{col_start}.jpg')
                cv2.imwrite(output_file,tile_cv_img)
                print(f'Saved Dected file at ({row_start}) as {output_file}')
            
            output_file=os.path.join('',f'{row_start}_{col_start}.jpg')
            cv2.imwrite(output_file,tile_cv_img)
            print(f'Saved file at ({row_start}) as {output_file}')

process_tiff(1024,100)
