# Welcome to CAPTCHA break tutorial !

# Imports
import cv2
import numpy as np
import os
import sys
import base64
import io
from imageio import imread
import matplotlib.pyplot as plt

# run on CPU, to run on GPU comment this line or write '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
from distutils.version import StrictVersion
from collections import defaultdict


# title of our window
title = "CAPTCHA"

# Env setup
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


# Model preparation 
PATH_TO_FROZEN_GRAPH = 'CAPTCHA_frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'CAPTCHA_labelmap.pbtxt'
NUM_CLASSES = 37


# Load a (frozen) Tensorflow model into memory.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# Detection
def Captcha_detection(image, average_distance_error=3):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Open image
            # print(image + "few")
               
            imagevcxvcx = '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'
            
            imgfewza = imread(io.BytesIO(base64.b64decode(str(image))))
            # plt.figure()
            # plt.imshow(imgfewza, cmap="gray")

            image_np = cv2.cvtColor(imgfewza, cv2.COLOR_BGR2RGB)
            image_np = cv2.resize(image_np, (0,0), fx=5, fy=5) 

            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Visualization of the results of a detection.
            (boxes, scores, classes, num_detections) = sess.run(
              [boxes, scores, classes, num_detections],
              feed_dict={image_tensor: image_np_expanded})
            vis_util.visualize_boxes_and_labels_on_image_array(
              image_np,
              np.squeeze(boxes),
              np.squeeze(classes).astype(np.int32),
              np.squeeze(scores),
              category_index,
              use_normalized_coordinates=True,
              line_thickness=2)
            # Show image with detection
            # cv2.imshow(title, cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
            # Save image with detection
            cv2.imwrite("Predicted_captcha.jpg", cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))


            # Bellow we do filtering stuff
            captcha_array = []
            # loop our all detection boxes
            for i,b in enumerate(boxes[0]):
                for Symbol in range(37):
                    if classes[0][i] == Symbol: # check if detected class equal to our symbols
                        if scores[0][i] >= 0.50: # do something only if detected score more han 0.65
                                            # x-left        # x-right
                            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2 # find x coordinates center of letter
                            # to captcha_array array save detected Symbol, middle X coordinates and detection percentage
                            captcha_array.append([category_index[Symbol].get('name'), mid_x, scores[0][i]])

            # rearange array acording to X coordinates datected
            for number in range(20):
                for captcha_number in range(len(captcha_array)-1):
                    if captcha_array[captcha_number][1] > captcha_array[captcha_number+1][1]:
                        temporary_captcha = captcha_array[captcha_number]
                        captcha_array[captcha_number] = captcha_array[captcha_number+1]
                        captcha_array[captcha_number+1] = temporary_captcha


            # Find average distance between detected symbols
            average = 0
            captcha_len = len(captcha_array)-1
            while captcha_len > 0:
                average += captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1]
                captcha_len -= 1
            # Increase average distance error
            average = average/(len(captcha_array)+average_distance_error)

            
            captcha_array_filtered = list(captcha_array)
            captcha_len = len(captcha_array)-1
            while captcha_len > 0:
                # if average distance is larger than error distance
                if captcha_array[captcha_len][1]- captcha_array[captcha_len-1][1] < average:
                    # check which symbol has higher detection percentage
                    if captcha_array[captcha_len][2] > captcha_array[captcha_len-1][2]:
                        del captcha_array_filtered[captcha_len-1]
                    else:
                        del captcha_array_filtered[captcha_len]
                captcha_len -= 1

            # Get final string from filtered CAPTCHA array
            captcha_string = ""
            for captcha_letter in range(len(captcha_array_filtered)):
                captcha_string += captcha_array_filtered[captcha_letter][0]
                
            return captcha_string
 
        

