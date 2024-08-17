# import sys
# print(sys.base_prefix)
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image
import os
import random
from scipy.interpolate import interp1d
import itertools
import colorsys
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import IPython.display

from Paths import ProjectPath

Path=ProjectPath()

def convertYOLO_bb_2_MRCNN_bb(Yolo_BB,height,width):
    output=[]
    for bb in Yolo_BB:
        x,y,w,h=bb
        x = x* width
        y= y * height
        w = w* width
        h = h * height

            # Convert center coordinates to top-left corner coordinates
        x1 = int(x - (w / 2))
        y1 = int(y - (h / 2))
        w = int(w)
        h = int(h)
        x2=x1+w
        y2=y1+h
        output.append([y1,x1,y2,x2])
    return np.array(output)
def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def display_instances(imgFile, imgPath,boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None, save_dir="",save=False):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances

    image=cv2.cvtColor(cv2.imread(os.path.join(imgPath,imgFile)),cv2.COLOR_BGR2RGB)
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if save:
        plt.savefig(os.path.join(save_dir,imgFile[:-3]+'png'))
    if auto_show:
        plt.show()

def display_instances_real_time(image, boxes, masks, class_ids, class_names,
                                scores=None, show_mask=True, show_bbox=True,
                                colors=None):
    """
    Return an image with drawn instances.
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
        return image
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # Generate random colors
    colors = colors or random_colors(N)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            cv2.rectangle(masked_image, (x1, y1), (x2, y2), color, 2)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = "{} {:.3f}".format(label, score) if score else label
        cv2.putText(masked_image, caption, (x1, y1 + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            verts = np.int32(verts)
            cv2.polylines(masked_image, [verts], isClosed=True, color=color, thickness=2)
    
    return masked_image.astype(np.uint8)




def visualise(masks,imgFile,imgPath,savePath=Path.SegmentatedImages_Path,save=False):
    """
    This function show masks over the given image
    INPUT:
        masks=masks [height,width,instances]
        imgFile: Image File name
        imgPath= Path of image file
        savePath= Path in which processed image to be saved
        save: whether to save processed image or not
    """

    img=cv2.imread(os.path.join(imgPath,imgFile))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)




    height,width,instances=masks.shape
    print(masks.shape)
    

    r,g,b=cv2.split(img)

    for i in range(instances):
        mask=masks[:,:,i]
        c1,c2,c3=random.randint(0,255),random.randint(0,255),random.randint(0,255)
        for k in range(height):
            for l in range(width):
                if mask[k][l]>0:
                    r[k][l]=c1
                    g[k][l]=c2
                    b[k][l]=c3
    
    img=cv2.merge([r,g,b])
    plt.imshow(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if save:
        cv2.imwrite(os.path.join(savePath,imgFile),img)
        print("Written File:",imgFile)


def visualise_with_boxes(masks,boxes,confidence,imgFile,imgPath,savePath=Path.SegmentatedImages_Path,save=False, Conf_threshold=0):
    """
    This function show masks with bounding boxes over the given image
    INPUT:
        masks=masks shape:[height,width,instances]
        boxes= normalised bounding boxes in yolo format (x,y,w,h) shape:[instances,4]
        confidence= confidence score for each bounding box shape:[instances]
        imgFile: Image File name
        imgPath= Path of image file
        savePath= Path in which processed image to be saved
        save: whether to save processed image or not
    """
    img=cv2.imread(os.path.join(imgPath,imgFile))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


    r,g,b=cv2.split(img)
    height,width,instances=masks.shape

    for i in range(instances):
        if confidence[i]>Conf_threshold:
            mask=masks[:,:,i]
            c1,c2,c3=random.randint(0,254),random.randint(0,254),random.randint(0,250)
            for k in range(height):
                for l in range(width):
                    if mask[k][l]>0:
                        r[k][l]=c1
                        g[k][l]=c2
                        b[k][l]=c3
    
    img=cv2.merge([r,g,b])
    count=0
    for i in range(len(confidence)):
        box=list(boxes[i])
        con=confidence[i]
        #print(str(con))
        
        if con>0:
            count+=1
            x,y,w,h=box
            img_height,img_width=img.shape[:2]
            x = x* img_width
            y= y * img_height
            width = w* img_width
            height = h * img_height

            # Convert center coordinates to top-left corner coordinates
            x = int(x - (width / 2))
            y = int(y - (height / 2))
            w = int(width)
            h = int(height)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255,255), 4)
            #cv2.putText(img, str(con), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            text = f"{con:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 6)
            
            # Draw the background rectangle
            cv2.rectangle(img, (x, y - text_height - baseline), (x + text_width, y), (255, 255, 255), cv2.FILLED)
            
            # Put the confidence text on the image
            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
    
    print(count)
    text = str(count)+" Grape Bunches Detected"
    x=y=0
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 6)
    cv2.rectangle(img, (x, y), (x + text_width+15, y+text_height+baseline+20), (255, 0, 0), cv2.FILLED)   
    cv2.putText(img, text, (x+15, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 4)
    plt.imshow(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if save:
        cv2.imwrite(os.path.join(savePath,imgFile),img)
        
        print("Written File:",imgFile)


def visualise_with_boxes2(masks,boxes,confidence,imgFile,imgPath,savePath=Path.SegmentatedImages_Path,save=False, Conf_threshold=0,resize=False):
    """
    This function show masks with bounding boxes over the given image
    INPUT:
        masks=masks shape:[height,width,instances]
        boxes= normalised bounding boxes in yolo format (x,y,w,h) shape:[instances,4]
        confidence= confidence score for each bounding box shape:[instances]
        imgFile: Image File name
        imgPath= Path of image file
        savePath= Path in which processed image to be saved
        save: whether to save processed image or not
    """
    img=cv2.imread(os.path.join(imgPath,imgFile))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    # if resize:
    #     imgheight,imgwidth,_=img.shape
    #     masks=resizeMasks(masks,imgheight,imgwidth)
    height,width,instances=masks.shape
    

    r,g,b=cv2.split(img)

    for i in range(instances):
        if confidence[i]>Conf_threshold:
            mask=masks[:,:,i]
            c1,c2,c3=random.randint(0,254),random.randint(0,254),random.randint(0,250)
            for k in range(height):
                for l in range(width):
                    if mask[k][l]>0:
                        r[k][l]=c1
                        g[k][l]=c2
                        b[k][l]=c3
    
    img=cv2.merge([r,g,b])
    count=0
    for i in range(len(confidence)):
        box=list(boxes[i])
        con=confidence[i]
        #print(str(con))
        
        if con>0:
            count+=1
            y1,x1,y2,x2=box
           
            
            # Convert center coordinates to top-left corner coordinates
            # x = int(x - (width / 2))
            # y = int(y - (height / 2))
            # w = int(width)
            # h = int(height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255,255), 4)
            #cv2.putText(img, str(con), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            text = f"{con:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 6)
            
            # Draw the background rectangle
            cv2.rectangle(img, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (255, 255, 255), cv2.FILLED)
            
            # Put the confidence text on the image
            cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 4)
    
    print(count)
    text = str(count)+" Grape Bunches Detected"
    x=y=0
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 6)
    cv2.rectangle(img, (x, y), (x + text_width+15, y+text_height+baseline+20), (255, 0, 0), cv2.FILLED)   
    cv2.putText(img, text, (x+15, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 4)
    plt.imshow(img)
    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    if save:
        cv2.imwrite(os.path.join(savePath,imgFile),img)
        
        print("Written File:",imgFile)

def convert_YoloMasks2MrcnnMasks(masks,height=1365,width=2048,numberOfMasks=1):
    """
    This function converts Masks in Yolo format to MRCNN format

    Input:
        masks= masks array shape:[instances,height,width]
        height=height of output masks
        width=width of output masks
        numberOfMasks=Number of masks in array
    """

    output_masks=np.zeros([height,width,numberOfMasks])

    for i in range(numberOfMasks):
        output_masks[:,:,i]=cv2.resize(masks[i,:,:],[width,height])

    return output_masks

def compute_mIOU(IOU_matrix):
    """
    This function calculates mean Intersection over Union
    Input:
        IOU_matrix: Either Matrix for IOUs of shape [PredictedInstances,GroundTruthInstances]  or 
                    Dictionary of IOU Matrix containing matrix for different images in batch
    """
    if type(IOU_matrix)==type({}):
        avg=0
        for key in IOU_matrix:
            arr=IOU_matrix[key].max(axis=1)
            avg+=arr.sum()/len(arr)
        miou=avg/len(IOU_matrix)
        return miou
    else:
        arr=IOU_matrix.max(axis=1)
        miou+=arr.sum()/len(arr)
        return miou
    

def compute_IOU_Matrix(masks1, masks2):
    """Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width, instances]
    """
    
    # If either set of masks is empty return empty result
    if masks1.shape[-1] == 0 or masks2.shape[-1] == 0:
        return np.zeros((masks1.shape[-1], masks2.shape[-1]))
    # flatten masks and compute their areas
    masks1 = np.reshape(masks1 > .5, (-1, masks1.shape[-1])).astype(np.float32)
    masks2 = np.reshape(masks2 > .5, (-1, masks2.shape[-1])).astype(np.float32)
    area1 = np.sum(masks1, axis=0)
    area2 = np.sum(masks2, axis=0)

    # intersections and union
    intersections = np.dot(masks1.T, masks2)
    union = area1[:, None] + area2[None, :] - intersections
    overlaps = intersections / union

    return overlaps

def trim_zeros(x):
    """It's common to have tensors larger than the available data and
    pad with zeros. This function removes rows that are all zeros.

    x: [rows, columns].
    """
    assert len(x.shape) == 2
    return x[~np.all(x == 0, axis=1)]

def compute_matches(gt_boxes, gt_class_ids, gt_masks,
                    pred_boxes, pred_class_ids, pred_scores, pred_masks,
                    iou_threshold=0.5, score_threshold=0.0,my_score_threshold=0.0):
    """Finds matches between prediction and ground truth instances.

    Returns:
        gt_match: 1-D array. For each GT box it has the index of the matched
                  predicted box.
        pred_match: 1-D array. For each predicted box, it has the index of
                    the matched ground truth box.
        overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Trim zero padding
    # TODO: cleaner to do zero unpadding upstream
    gt_boxes = trim_zeros(gt_boxes)
    gt_masks = gt_masks[..., :gt_boxes.shape[0]]
    pred_boxes = trim_zeros(pred_boxes)
    pred_scores = pred_scores[:pred_boxes.shape[0]]

   # print('predscores1',list(pred_scores))
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]
    low_score_indx=np.where(pred_scores[indices]<my_score_threshold)[0]
    
    if low_score_indx.size > 0:
        indices=indices[:low_score_indx[0]]
    pred_boxes = pred_boxes[indices]
    pred_class_ids = pred_class_ids[indices]
    pred_scores = pred_scores[indices]
    pred_masks = pred_masks[..., indices]
    #print('predscores1',list(pred_scores))

    # Compute IoU overlaps [pred_masks, gt_masks]
    overlaps = compute_IOU_Matrix(pred_masks, gt_masks)

    # Loop through predictions and find matching ground truth boxes
    match_count = 0
    pred_match = -1 * np.ones([pred_boxes.shape[0]])
    gt_match = -1 * np.ones([gt_boxes.shape[0]])
    
    for i in range(len(pred_boxes)):
        # Find best matching ground truth box
        # 1. Sort matches by score
        sorted_ixs = np.argsort(overlaps[i])[::-1]
        # 2. Remove low scores
        low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
        if low_score_idx.size > 0:
            sorted_ixs = sorted_ixs[:low_score_idx[0]]
       
        # 3. Find the match
        for j in sorted_ixs:
            # If ground truth box is already matched, go to next one
            if gt_match[j] > -1:
                continue
            # If we reach IoU smaller than the threshold, end the loop
            iou = overlaps[i, j]
            if iou < iou_threshold:
                break
            # Do we have a match?
            if pred_class_ids[i] == gt_class_ids[j]:
                match_count += 1
                gt_match[j] = i
                pred_match[i] = j
                break

    return gt_match, pred_match, overlaps


def compute_ap_mrcnn(gt_boxes, gt_class_ids, gt_masks,
               pred_boxes, pred_class_ids, pred_scores, pred_masks,
               iou_threshold=0.5,score_threshold=0.0,my_score_threshold=0.0):
    """Compute Average Precision at a set IoU threshold (default 0.5).

    Returns:
    mAP: Mean Average Precision
    precisions: List of precisions at different class score thresholds.
    recalls: List of recall values at different class score thresholds.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
    """
    # Get matches and overlaps
    gt_match, pred_match, overlaps = compute_matches(
        gt_boxes, gt_class_ids, gt_masks,
        pred_boxes, pred_class_ids, pred_scores, pred_masks,
        iou_threshold,score_threshold=score_threshold,my_score_threshold=my_score_threshold)

    # Compute precision and recall at each prediction box step
    pre_org=precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
    recall_org=recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

    # Pad with start and end values to simplify the math
    precisions = np.concatenate([[0], precisions, [0]])
    recalls = np.concatenate([[0], recalls, [1]])

    # Ensure precision values decrease but don't increase. This way, the
    # precision value at each recall threshold is the maximum it can be
    # for all following recall thresholds, as specified by the VOC paper.
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = np.maximum(precisions[i], precisions[i + 1])

    # Compute mean AP over recall range
    indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
    mAP = np.sum((recalls[indices] - recalls[indices - 1]) *
                 precisions[indices])

    return mAP, precisions, recalls, overlaps, pre_org,recall_org


def compute_ap_yolov8(recall, precision, method = "interp" ):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Args:
        recall (list): The recall curve.
        precision (list): The precision curve.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
     # methods: 'continuous', 'interp'
    if method == "interp":
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x-axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec

def convert_scores(scores):
    output=[]
    for score in scores:
        
        if score>=0.9999:
            
             x=score*10000
             x=x-int(x)
             score=x
             score=(score*0.8)+0.2
        output.append(score)
    return np.array(output)

def plot_PR_curve(precisions_and_recalls,mAP,save=False, save_dir=Path.ResultCurvesPath, saveFileName="PR_Curve.png",in_between_values=10000):
    """
    This function plots PR curve.

    precisions and recalls: A dictionary containing precision and recall arrays for each image in batch
                            {image:{precisions:[],recalls:[]}}
    mAP: mean AP for batch

    save: whether to save plot

    save_dir=which dir to save plot

    saveFileName=name by which plot needs to be saved

    in_between_values= Value distribution for interpolation
    """

    px=np.linspace(0,1,in_between_values)
    interpolated_ys=[]

    for key in precisions_and_recalls:
        precisions=precisions_and_recalls[key]['precisions']
        recalls=precisions_and_recalls[key]['recalls']
        f = interp1d(recalls,precisions, kind='linear', fill_value='interpolate')
        interpolated_y = f(px)
        interpolated_ys.append(interpolated_y)

    py = np.mean(interpolated_ys, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=3, color="blue", label="Grapes %.3f mAP@0.5" % mAP)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve for Grape Instance Segmentation")
    if save:
        save_dir=os.path.join(save_dir,saveFileName)
        fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)
    # if on_plot:
    #     on_plot(save_dir)



def plot_PrecisionConfidence_curve(scores,precisions,averagePrecisiion=0,save=False, save_dir=Path.ResultCurvesPath, saveFileName="PC_Curve.png",in_between_values=10000):
    """
    This function plots Precision-Confidence curve.

    scores: confidence array
    precisions: A dictionary containing precision array for each image in batch
                            
    averagePrecision: mean Precision for batch at confidence threshold 0.5

    save: whether to save plot

    save_dir=which dir to save plot

    saveFileName=name by which plot needs to be saved

     in_between_values= Value distribution for interpolation
    """

    px=np.linspace(0,1,in_between_values)
    interpolated_ys=[]

    for key in precisions:
        f = interp1d(scores,precisions[key], kind='linear', fill_value='extrapolate')
        interpolated_y = f(px)
        interpolated_ys.append(interpolated_y)

    py = np.mean(interpolated_ys, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=3, color="blue", label="Grapes %.3f AveragePrecision@0.5" % averagePrecisiion)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Confidence Curve for Grape Instance Segmentation")
    if save:
        save_dir=os.path.join(save_dir,saveFileName)
        fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)
    # if on_plot:
    #     on_plot(save_dir)


def plot_RecallConfidence_curve(scores,recalls,averageRecall=0,save=False, save_dir=Path.ResultCurvesPath, saveFileName="RC_Curve.png",in_between_values=10000):
    """
    This function plots Recall-Confidence curve.

    scores: confidence array
    recalls: A dictionary containing recall array for each image in batch
                            
    averageRecall: mean Recall for batch at confidence threshold 0.5

    save: whether to save plot

    save_dir=which dir to save plot

    saveFileName=name by which plot needs to be saved

     in_between_values= Value distribution for interpolation
    """

    px=np.linspace(0,1,in_between_values)
    interpolated_ys=[]

    for key in recalls:
        f = interp1d(scores,recalls[key], kind='linear', fill_value='extrapolate')
        interpolated_y = f(px)
        interpolated_ys.append(interpolated_y)
    # print("hello")
    py = np.mean(interpolated_ys, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=3, color="blue", label="Grapes %.3f AverageRecall@0.5" % averageRecall)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Recall-Confidence Curve for Grape Instance Segmentation")
    if save:
        save_dir=os.path.join(save_dir,saveFileName)
        fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)
    # if on_plot:
    #     on_plot(save_dir)


def plot_F1ScoreConfidence_curve(scores,F1Scores,averageF1=0,save=False, save_dir=Path.ResultCurvesPath, saveFileName="FC_Curve.png",in_between_values=10000):
    """
    This function plots F1-score-Confidence curve.

    scores: confidence array
    F1Scores: A dictionary containing F1-score array for each image in batch
                            
    averageF1: mean F1-Score for batch at confidence threshold 0.5

    save: whether to save plot

    save_dir=which dir to save plot

    saveFileName=name by which plot needs to be saved

     in_between_values= Value distribution for interpolation
    """

    px=np.linspace(0,1,in_between_values)
    interpolated_ys=[]

    for key in F1Scores:
        f = interp1d(scores,F1Scores[key], kind='linear', fill_value='interpolate')
        interpolated_y = f(px)
        interpolated_ys.append(interpolated_y)

    py = np.mean(interpolated_ys, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    ax.plot(px, py, linewidth=3, color="blue", label="Grapes %.3f Average_F1-Score@0.5" % averageF1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1-Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("F1_Score-Confidence Curve for Grape Instance Segmentation")
    if save:
        save_dir=os.path.join(save_dir,saveFileName)
        fig.savefig(save_dir, dpi=250)
    plt.show()
    plt.close(fig)
    # if on_plot:
    #     on_plot(save_dir)

