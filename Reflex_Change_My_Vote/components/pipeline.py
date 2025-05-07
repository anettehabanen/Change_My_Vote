
import numpy as np
import math
import time

from PIL import Image
import albumentations as A
import cv2
from moviepy import VideoFileClip, AudioFileClip
import imageio.v3 as iio

import os
import os.path as osp
import shutil
from tqdm import tqdm

from ultralytics import YOLO

import subprocess


### Saving the video as frames ###

def save_video_as_frames(video, video_frames_path, id_length):

    if not os.path.exists(video_frames_path):
        os.makedirs(video_frames_path)

    pbar = tqdm()
    
    vidcap = cv2.VideoCapture(video)
    success,image = vidcap.read()
    count = 0
    while success:
        image_id = '0' * (id_length - len(str(count))) + str(count)
        frame_path = osp.join(video_frames_path, 'frame_{}.png'.format(image_id))
        cv2.imwrite(frame_path, image)
        success,image = vidcap.read()
        count += 1
        pbar.update(1)

    pbar.close()



### Corner coordinates ###

def read_in_coordinates(file_path):

    coordinates = []
    file = open(file_path,'r')
    while True:
        content=file.readline()
        if not content:
            break
        coordinates = [ int(float(x)) for x in content.strip().split(" ") ]
    file.close()

    return coordinates


def save_corner_coordinates(coordinates, labels_path, file_name):

    file_path = os.path.join(labels_path, file_name)

    with open(file_path, "w") as file:
        file.write(str(coordinates[0][0][0]) + " " + str(coordinates[0][0][1]) + " " + 
                   str(coordinates[1][0][0]) + " " + str(coordinates[1][0][1]) + " " + 
                   str(coordinates[2][0][0]) + " " + str(coordinates[2][0][1]) + " " + 
                   str(coordinates[3][0][0]) + " " + str(coordinates[3][0][1]))
        file.close()  



def lucas_kanade_video(input_video_frames: str, labels_path: str, startFeatures):

    # Find video frames
    frames = sorted([file for file in os.listdir(input_video_frames) if file.split(".")[1] == "png"])

    # First frame
    frame_path = os.path.join(input_video_frames, frames[0])
    old_frame = cv2.imread(frame_path)
    frame_width = old_frame.shape[1]
    frame_height = old_frame.shape[0]

    # Initial corners
    old_frame_gs = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY) # Grayscale
    mask_linestring = np.zeros_like(old_frame, dtype=np.uint8) # create a mask for optical flow
    p0 = startFeatures # initial corners
    save_corner_coordinates(p0, labels_path, frames[0].split('.')[0]+str(".txt"))

    # Going through all the frames
    for i in tqdm(range(1, len(frames))):
        frame_path = os.path.join(input_video_frames, frames[i])
        new_frame = cv2.imread(frame_path)
        new_frame_gs = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        '''
        p1, st, err = cv2.calcOpticalFlowPyrLK(prevImg = old_frame_gs, 
                                               nextImg = new_frame_gs, 
                                               prevPts = p0, 
                                               nextPts = p0, 
                                               winSize = (3,3),)
        '''
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame_gs, new_frame_gs, p0, None, (3,3))
        save_corner_coordinates(p1, labels_path, frames[i].split('.')[0]+str(".txt"))

        # make new frame the old one
        old_frame_gs = new_frame_gs.copy()
        p0 = p1.reshape(-1,1,2)
    
    return




### Using YOLO to find digits ###

def tracking_with_bounding_boxes(model, video_frames_path, video_bboxes_path, tracked_labels_path, add_to_boxes_width, add_to_boxes_height):

    first_3_not_found = True
    first_3_index = 0

    if not os.path.exists(video_bboxes_path):
        os.makedirs(video_bboxes_path)

    frames = [img for img in os.listdir(video_frames_path) if img.endswith(".png")]
    frames.sort()

    for i in range(len(frames)):
        frame_path = osp.join(video_frames_path, frames[i])
        frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        height, width, channels = frame.shape
    
        results = model.track(frame, 
                              imgsz = 1280,
                              conf = 0.2,
                              iou = 0.8,
                              persist=True)

        classes = np.array(results[0].boxes.cls.cpu())
        bboxes = np.array(results[0].boxes.xywhn.cpu())
        frame_labels = np.array([np.insert(bboxes[i], 0, classes[i]) for i in range(len(bboxes))])

        
        # Reading in box corner info
        labels_path = os.path.join(tracked_labels_path, frames[i].replace("png", "txt"))
        corner_labels = read_in_coordinates(labels_path)

        x_coord = [corner_labels[j] for j in range(0, len(corner_labels), 2)]
        y_coord = [corner_labels[j+1] for j in range(0, len(corner_labels)-1, 2)]
        xmin, xmax, ymin, ymax = min(x_coord)/width, max(x_coord)/width, min(y_coord)/height, max(y_coord)/height

        
        # Going through and filtering YOLO results
        filtered_labels = []
        for j in range(len(frame_labels)):

            bounding_box = frame_labels[j]
            if bounding_box[1] > xmin and bounding_box[1] < xmax and bounding_box[2] > ymin and bounding_box[2] < ymax:

                if len(filtered_labels) == 0:
                    filtered_labels.append(frame_labels[j]) 
                else:
                    new_object = True
                    for k in range(j):
                        diffrence = abs(frame_labels[j][1] - frame_labels[k][1])
                        if diffrence <= 0.001:
                            new_object = False
                            break
                    if new_object: # Only adding this bounding box if it actually is a new digit and not very close prediction to already existing digit
                        filtered_labels.append(frame_labels[j])

        
        if len(filtered_labels) == 3 and first_3_not_found: # Finding the first frame where we have found all 3 digits 
            first_3_index = i
            first_3_not_found = False

        found_bboxes = min(len(filtered_labels), 3) # Only saving the first 3 digits

        labels_file_name = frames[i].replace("frame", "bboxes").replace("png", "txt")
        labels_path = osp.join(video_bboxes_path, labels_file_name)

        add_width = add_to_boxes_width / width
        add_height = add_to_boxes_height / height
        
        # write the label and bounding boxes
        if found_bboxes == 0:
            with open(labels_path, "w") as file:
                pass 
        else:
            with open(labels_path, "w") as file:
                for j in range(found_bboxes):
                    file.write(str(int(filtered_labels[j][0])) + " " +
                           str(float(filtered_labels[j][1])) + " " + 
                           str(float(filtered_labels[j][2])) + " " + 
                           str(float(filtered_labels[j][3]) + add_width) + " " + 
                           str(float(filtered_labels[j][4]) + add_height) + "\n")
                file.close()  


    return first_3_index





### Reading in and saving label info ###

def reading_in_bbox_txt_file(bbox_path) :
    bboxes = []
    file = open(bbox_path,'r')
    while True:
        content=file.readline()
        if not content:
            break
        elements = [ float(x) for x in content.strip().split(" ") ]
        bboxes.append(elements)
    file.close()

    if len(bboxes) != 0:
        sorted_indices = np.argsort(np.array(bboxes)[:, 1]) # Based on xcenter
        return np.array(bboxes)[sorted_indices]
    else:
        return bboxes
    

def save_bboxes_into_txt(path, bboxes):
    if len(bboxes) == 0:
        with open(path, "w") as file:
            pass 
    else:
        with open(path, "w") as file:
            for j in range(len(bboxes)):
                if len(bboxes[j]) == 0:
                    continue
                else:
                    file.write(str(int(bboxes[j][0])) + " " +
                           str(float(bboxes[j][1])) + " " + 
                           str(float(bboxes[j][2])) + " " + 
                           str(float(bboxes[j][3])) + " " + 
                           str(float(bboxes[j][4])) + "\n")
            file.close() 



### Creating frames algorithmically ###

def creating_missing_bboxes(yolo_boxes_path, bboxes_path, first_3_idx, id_length):

    bboxes_files = [f for f in os.listdir(yolo_boxes_path) if f.endswith('.txt')]
    bboxes_files.sort()
    bboxes_3_labels = []
    last_x = [0, 0, 0]
    prediction_next = [0, 0, 0]

    if not os.path.exists(bboxes_path):
        os.makedirs(bboxes_path)


    print("Creating 3 bbox spaces for every frame")
    for i in tqdm(range(len(bboxes_files))):

        frame_label_path = osp.join(yolo_boxes_path, bboxes_files[i])
        frame_bboxes = reading_in_bbox_txt_file(frame_label_path)
    
        if i < first_3_idx: # Before the first 3-digit frame, just save the bounding box info
            bboxes_3_labels.append(frame_bboxes)
        elif i == first_3_idx: # Save info about the 3-digits
            bboxes_3_labels.append(frame_bboxes)
            last_x = [frame_bboxes[0][1], frame_bboxes[1][1], frame_bboxes[2][1]]
            prediction_next = [frame_bboxes[0][1], frame_bboxes[1][1], frame_bboxes[2][1]]
        else:
            if len(frame_bboxes) == 0: # Frame empty, but save 3 arrays
                bboxes_3_labels.append([[], [], []])
            elif len(frame_bboxes) == 3: # All digits found
                bboxes_3_labels.append(frame_bboxes)
                prediction_next = [frame_bboxes[0][1] + (frame_bboxes[0][1] - last_x[0]), 
                                   frame_bboxes[1][1] + (frame_bboxes[1][1] - last_x[1]), 
                                   frame_bboxes[2][1] + (frame_bboxes[2][1] - last_x[2]) ]
                last_x = [frame_bboxes[0][1],frame_bboxes[1][1], frame_bboxes[2][1] ]
            else: # 1 or 2 digits found
                new_bbox = [[], [], []]
                prediction = [prediction_next[0] + (prediction_next[0] - last_x[0]),
                              prediction_next[1] + (prediction_next[1] - last_x[1]),
                              prediction_next[2] + (prediction_next[2] - last_x[2]),
                    
                ]
                for j in range(len(frame_bboxes)): # Find to which digit this bounding box info belongs to
                    digit = frame_bboxes[j]
                    differences = abs(prediction_next - frame_bboxes[j][1])
                    min_index = min(range(len(differences)), key=differences.__getitem__)
                    new_bbox[min_index] = digit
                    prediction[min_index] = digit[1] + (digit[1] - last_x[min_index])
                    last_x[min_index] = digit[1]
                prediction_next = prediction
                bboxes_3_labels.append(new_bbox)


    print("Creating missing bboxes")
    for i in range(3): # Going through every digit object
        pbar = tqdm(total = len(bboxes_3_labels)-1)

        j = first_3_idx
        while j < len(bboxes_3_labels)-1:
            pbar.update(1)

            digit = bboxes_3_labels[j][i] # Starting digit
            for k in range(j+1, len(bboxes_3_labels)): # Going through next frames
                if len(bboxes_3_labels[k][i]) == 0: # Model didn't find digit from this frame
                    if k >= len(bboxes_3_labels)-1:
                        j = k
                        break
                    continue
                else: # (some) next frame has this digit
                    if j + 1 == k: # We found digit from next frame
                        j += 1
                        break
                    elif k >= len(bboxes_3_labels)-1:
                        j = k
                        break
                    else: # We found digit somewhere further away
                        if k-j < 20:
                            # We generate bboxes
                            generate_frames_nr = k-j-1
                            for m in range(1, generate_frames_nr + 1):
                                new_frame = [digit[0],
                                            digit[1] + ((bboxes_3_labels[k][i][1] - digit[1]) / generate_frames_nr * m),
                                            digit[2] + ((bboxes_3_labels[k][i][2] - digit[2]) / generate_frames_nr * m),
                                            digit[3] + ((bboxes_3_labels[k][i][3] - digit[3]) / generate_frames_nr * m),
                                            digit[4] + ((bboxes_3_labels[k][i][4] - digit[4]) / generate_frames_nr * m)
                                            ]
                                bboxes_3_labels[j+m][i] = new_frame
                        j = k
                        break

    # Saving new bounding boxes
    for i in range(len(bboxes_3_labels)):
        image_id = '0' * (id_length - len(str(i))) + str(i)
        labels_path = osp.join(bboxes_path, "bboxes_{}.txt".format(image_id))
        save_bboxes_into_txt(labels_path, bboxes_3_labels[i])




### Getting digit from the frame ### 

def finding_digits_from_the_frame(frame, frame_width, frame_height, frame_labels, padding_up_down, padding_sides):
    
    digits = []
    
    for i in range(len(frame_labels)):
        digit_width = int(frame_labels[i][3] * frame_width) + padding_sides # Adding more pixels for a boundary
        digit_height = int(frame_labels[i][4] * frame_height) + padding_up_down
        xmin = int((frame_labels[i][1] * frame_width) - digit_width/2)
        ymin = int((frame_labels[i][2] * frame_height) - digit_height/2)
    
        digit = frame[ymin:ymin+digit_height, xmin:xmin+digit_width]
        digits.append(digit)

    return digits



### Making the digits into a squares (aka creating more background) ###

def changing_digit_into_square(digits, square_digits_path, frame_id, padding_up_down, padding_sides):

    if not os.path.exists(square_digits_path):
        os.makedirs(square_digits_path)
    if not os.path.exists(osp.join(square_digits_path, "0")):
        os.makedirs(osp.join(square_digits_path, "0"))
    if not os.path.exists(osp.join(square_digits_path, "1")):
        os.makedirs(osp.join(square_digits_path, "1"))
    if not os.path.exists(osp.join(square_digits_path, "2")):
        os.makedirs(osp.join(square_digits_path, "2"))
        
    
    for i in range(len(digits)):
    
        digit = digits[i]
        digit_width, digit_height = digit.shape[1], digit.shape[0]
        square_hw = max(digit_width, digit_height)
        square = np.zeros((square_hw, square_hw, 3), dtype=int)
    
        bigger_height = True if digit_width <= digit_height else False
        mask_xmin, mask_ymin, mask_width, mask_height = 0, 0, 0, 0

        # When the digit is tall and we have to extend the sides.
        if bigger_height:
            small_bg = digit[0:digit_height, 0:10]
            for j in range(0, square_hw, 10):
                if math.floor(square_hw/10)*10 == j: # last part to cover 
                    new_bg_width = square_hw%10 
                    small_bg = digit[0:digit_height, 0:new_bg_width]
                    square[0:digit_height, j:j+new_bg_width] = small_bg
                else:
                    square[0:digit_height, j:j+10] = small_bg
            xmin = int((square_hw-digit_width)/2)
            square[0:square_hw, xmin:xmin+digit_width] = digit

            mask_xmin = int(xmin / square_hw * 255)
            mask_ymin = int((padding_up_down/2) / square_hw * 255)
            mask_width = int((xmin+digit_width) / square_hw * 255)
            mask_height = int((square_hw-(padding_up_down//2)) / square_hw * 255)
        
        else:  # When the digit is wide and we have to extend the up and down part.
            small_bg = digit[0:10, 0:digit_width]
            for j in range(0, square_hw, 10):
                if math.floor(square_hw/10)*10 == j:
                    new_bg_height = square_hw%10 
                    small_bg = digit[0:new_bg_height, 0:digit_width]
                    square[j:j+new_bg_height, 0:digit_width] = small_bg
                else:
                    square[j:j+10, 0:digit_width] = small_bg
            ymin = int((square_hw-digit_height)/2)
            square[ymin:ymin+digit_height, 0:square_hw] = digit

            mask_xmin = int((padding_sides//2) / square_hw * 255)
            mask_ymin = int(ymin / square_hw * 255)
            mask_width = int((square_hw-(padding_sides//2)) / square_hw * 255)
            mask_height = int((ymin+digit_height) / square_hw * 255)

        # Saving square image
        output_image = Image.fromarray(np.uint8(square)).convert('RGB').resize((256,256))
        image_path = osp.join(square_digits_path, "{}/{}_{}.png".format(i, frame_id, i))
        output_image.save(image_path, format='png')

        # Saving a mask
        mask = np.zeros((256, 256), np.float32)
        mask[mask_ymin:mask_height, mask_xmin:mask_width] = 1
        mask_path = osp.join(square_digits_path, "{}/{}_{}_mask.png".format(i, frame_id, i))
        mask = Image.fromarray(np.uint8(mask*255)).convert('L')
        mask.save(mask_path, format='png')



### WavePaint image generation ###

def wavepaint_predictions(square_digits_path, wavepaint_predict, wavepaint_weights):

    generated_img = osp.join(square_digits_path, "generated") 
    masked_img = osp.join(square_digits_path, "masked")

    if not os.path.exists(generated_img):
        os.makedirs(generated_img)

    if not os.path.exists(masked_img):
        os.makedirs(masked_img)

    print("Calling the wavepaint model")

    os.system('python %s -model_path %s -test_data %s -generated_img %s -masked_img %s' % (wavepaint_predict, wavepaint_weights, square_digits_path+"/0", generated_img, masked_img))
    os.system('python %s -model_path %s -test_data %s -generated_img %s -masked_img %s' % (wavepaint_predict, wavepaint_weights, square_digits_path+"/1", generated_img, masked_img))
    os.system('python %s -model_path %s -test_data %s -generated_img %s -masked_img %s' % (wavepaint_predict, wavepaint_weights, square_digits_path+"/2", generated_img, masked_img))
 
    print("After predictions")
    
    results_file_names = [f for f in os.listdir(generated_img) if f.endswith('.png')]
    results_file_names.sort()

    # Full paths for generated images
    predictions = []
    for i in range(len(results_file_names)):
        if results_file_names[i][-4:] == '.png':
            predictions.append(osp.join(generated_img, results_file_names[i]))

    return predictions




def generating_new_digits(predictions, frame_labels, frame_width, frame_height, padding_up_down, padding_sides):

    new_digits = []
    for i in range(len(predictions)):

        prediction = cv2.cvtColor(cv2.imread(predictions[i]), cv2.COLOR_BGR2RGB)

        # Here we have to find the digit sizes
        org_digit_width = int(frame_labels[i][3] * frame_width + padding_sides)
        org_digit_height = int(frame_labels[i][4] * frame_height + padding_up_down)

        bigger_height = True if org_digit_width <= org_digit_height else False
        if bigger_height:
            image = cv2.resize(prediction, (org_digit_height, org_digit_height), interpolation=cv2.INTER_CUBIC)
            side = int((org_digit_height - org_digit_width)/2)
            digit = image[ : , side:side+org_digit_width]
        else:
            image = cv2.resize(prediction, (org_digit_width, org_digit_width), interpolation=cv2.INTER_CUBIC)
            side = int((org_digit_width - org_digit_height)/2)
            digit = image[side:side+org_digit_height , :]
    
        new_digits.append(digit)
        i += 1
    return new_digits




### Placing the predictions into frame ####


def replacing_digits_with_predictions(image_replaced, frame_labels, frame_width, frame_height, generated_digits, changed_frames_path, image_id, padding_up_down, padding_sides):
    
    for i in range(len(generated_digits)):
        # Finding the position
        digit_width = int(frame_labels[i][3] * frame_width + padding_sides) # Adding more pixels for a boundary
        digit_height = int(frame_labels[i][4] * frame_height + padding_up_down )
        xmin = int((frame_labels[i][1] * frame_width) - digit_width/2)
        ymin = int((frame_labels[i][2] * frame_height) - digit_height/2)
    
        # Change digit
        digit = (generated_digits[i]).astype('int')
        rgba = np.dstack((digit, np.full(digit.shape[:-1], 255)))
        digit = Image.fromarray(np.uint8(rgba)).convert('RGBA')
    
        # Placing digit onto frame
        image_replaced.paste(digit, (xmin, ymin), digit)

    return image_replaced



def changing_digits_in_the_frames(frames_path, 
                                  bboxes_path, 
                                  square_digits_path, 
                                  changed_frames_path, 
                                  id_length, 
                                  padding_up_down, 
                                  padding_sides, 
                                  wavepaint_predict, 
                                  wavepaint_weights,
                                  tracked_labels_path,
                                  new_digit):

    frames_files = [f for f in os.listdir(frames_path) if f.endswith('.png')]
    frames_files.sort()
    
    bboxes_files = [f for f in os.listdir(bboxes_path) if f.endswith('.txt')]
    bboxes_files.sort()

    if not os.path.exists(changed_frames_path):
        os.makedirs(changed_frames_path)

    # Finding frame size
    frame_1_path = osp.join(frames_path, frames_files[0])
    frame_1 = cv2.cvtColor(cv2.imread(frame_1_path), cv2.COLOR_BGR2RGB)
    frame_width = frame_1.shape[1]
    frame_height = frame_1.shape[0]

    
    print("Creating square digits")

    for idx in tqdm(range(len(frames_files))):
        
        frame_path = osp.join(frames_path, frames_files[idx])
        frame_np = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB)
        label_path = osp.join(bboxes_path, bboxes_files[idx])
        frame_labels = reading_in_bbox_txt_file(label_path)
        image_id = '0' * (id_length - len(str(idx))) + str(idx)

        # Getting the digit from frame
        digits = finding_digits_from_the_frame(frame_np, frame_width, frame_height, frame_labels, padding_up_down, padding_sides)

        if len(digits) != 0:

            # Adding background to the digit
            changing_digit_into_square(digits, square_digits_path, image_id, padding_up_down, padding_sides) # Saving all the squared digits


    
    print("WavePaint making predictions")
    # Generating results
    prediction_files = wavepaint_predictions(square_digits_path, wavepaint_predict, wavepaint_weights)
    prediction_files.sort()

    prediction_idx = 0

    print("Placing predictions and new digit onto frames")
    # Placing results into a video

    for idx in tqdm(range(len(frames_files))):
        frame_path = osp.join(frames_path, frames_files[idx])
        frame_image = Image.open(frame_path).convert('RGBA')
        label_path = osp.join(bboxes_path, bboxes_files[idx])
        frame_labels = reading_in_bbox_txt_file(label_path)
        image_id = '0' * (id_length - len(str(idx))) + str(idx)

        # Is there a digit to change?
        nr_of_digits = len(frame_labels)

        if nr_of_digits != 0:
            frame_digit_predictions = prediction_files[prediction_idx:prediction_idx+nr_of_digits]

            # Formatting the new digits
            generated_digits = generating_new_digits(frame_digit_predictions, frame_labels, frame_width, frame_height, padding_up_down, padding_sides)

            # Placing digits into the frame and saving it
            frame_image = replacing_digits_with_predictions(frame_image, frame_labels, frame_width, frame_height, generated_digits, changed_frames_path, image_id, padding_up_down, padding_sides)
            
            prediction_idx += nr_of_digits


        # Adding new digits onto frame
        corner_file_path = os.path.join(tracked_labels_path, frames_files[idx].split('.')[0]+str(".txt"))
        final_frame = placing_new_digit_into_frame(frame_image, new_digit, corner_file_path)

            
        # Saving the new frame
        file_path = osp.join(changed_frames_path, "frame_{}_r.png".format(image_id))
        final_frame_resize = final_frame.resize((1600, 900))
        final_frame_resize.save(file_path, format='png')



### Processing the new 3-digit code image ###


def remove_background(image):

    new_image = np.zeros((image.shape[0], image.shape[1], 4))
    
    for height in range(len(image)):
        for width in range(len(image[height])):
            pixel = image[height][width]
            if pixel[0] >= 180 and pixel[1] >= 180 and pixel[2] >= 180:
                new_image[height][width] = [255, 255, 255, 0]
            else:
                new_image[height][width] = [pixel[0], pixel[1], pixel[2], 255]
                new_image[height][width] = [max(pixel[0]-50, 0), 
                                            max(pixel[1]-50, 0), 
                                            max(pixel[2]-50, 0), 
                                            255]
                                            

    return new_image


def blur_image():
    return A.Compose([
        A.Blur(p=1, blur_limit=(5, 5))
    ])
bluring = blur_image()


def prepare_new_code_image(new_digit_path, xmin, xmax, ymin, ymax):
    image = cv2.cvtColor(cv2.imread(new_digit_path), cv2.COLOR_BGR2RGB)
    cropped = image[ymin:ymax, xmin:xmax]
    blured_digit = bluring(image=np.array(cropped))['image'] 
    white_bg = remove_background(blured_digit)
    return white_bg



def resize_image(h, w):
    return A.Compose([
        A.Resize(p=1, height=h, width=w)
    ])
        



def image_perspective_transform(digit, digit_width, digit_height, x_coord, y_coord, xmin, ymin):
    
    org_corners = np.float32([[0,0],[digit_width, 0], [0, digit_height],[digit_width, digit_height]])
    new_corners = np.float32([[x_coord[0] - xmin, y_coord[0] - ymin],
                              [x_coord[1] - xmin, y_coord[1] - ymin],
                              [x_coord[2] - xmin, y_coord[2] - ymin],
                              [x_coord[3] - xmin, y_coord[3] - ymin]])

    M = cv2.getPerspectiveTransform(org_corners,new_corners)
    transformed = cv2.warpPerspective(digit, M, (digit_width, digit_height),flags=cv2.INTER_LINEAR)

    return transformed



def placing_new_digit_into_frame(frame, new_digit, corner_file_path):
    
    corners = read_in_coordinates(corner_file_path)

    x_coord = []
    y_coord = []
    for j in range(0, len(corners), 2):
        x_coord.append(corners[j])
        y_coord.append(corners[j+1])

    xmin, xmax = min(x_coord), max(x_coord)
    ymin, ymax = min(y_coord), max(y_coord)

    digit_width, digit_height = xmax - xmin, ymax - ymin

    resizing = resize_image(digit_height, digit_width)
    resized_digit = resizing(image=np.array(new_digit))['image']

    transformed_digit = image_perspective_transform(resized_digit, digit_width, digit_height, x_coord, y_coord, xmin, ymin)
    digit = Image.fromarray(np.uint8(transformed_digit)).convert('RGBA')

    # Place new digit into the frame
    frame.paste(digit, (xmin, ymin), digit)

    return frame






### Putting the video together ###



def get_fps(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps


def creating_video_from_frames(frames_folder, video_name, fps):

    images = [img for img in os.listdir(frames_folder) if img.endswith(".png")]
    images.sort(key=lambda x: (int(x.split("_")[-2])))
    frame = cv2.imread(os.path.join(frames_folder, images[0]))
    height, width, layers = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, fps, (width,height))
    
    for image in tqdm(images):
        video.write(cv2.imread(os.path.join(frames_folder, image)))
    video.release()


def getting_video_audio(video_file, audio_file):
    
    video_clip = VideoFileClip(video_file)  # Load the video clip
    audio_clip = video_clip.audio # Extract the audio from the video clip
    audio_clip.write_audiofile(audio_file)  # Write the audio to a separate file
    
    audio_clip.close()
    video_clip.close()


def adding_audio_to_a_video(video_file, audio_file, video_output):
    audio = AudioFileClip(audio_file)
    video = VideoFileClip(video_file)
    video.audio = audio
    video.write_videofile(video_output)


def remove_files(path):
    shutil.rmtree(path, ignore_errors=True)
