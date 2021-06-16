import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np

sys.path.append('./tensorflow1/')
sys.path.append('./tensorflow1/research')
sys.path.append('./tensorflow1/research/slim')

from object_detection.utils import label_map_util

PATH_TO_FROZEN_GRAPH = './graphs/frozen_inference_graph.pb'
PATH_TO_LABELS = './graphs/labelmap.pbtxt'
NUM_CLASSES = 37

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                config = tf.ConfigProto()
                #config.gpu_options.allow_growth = True
                #config.gpu_options.per_process_gpu_memory_fraction = 0.4
                #config.log_device_placement = False
                config.intra_op_parallelism_threads = 0
                config.inter_op_parallelism_threads = 2
                config.allow_soft_placement=True
                sess = tf.Session(config=config, graph=detection_graph)
                return sess, detection_graph

def inference(sess, detection_graph, img_arr, average_distance_error=3):
        # print(img_arr)
        image_np_expanded = np.expand_dims(img_arr, axis=0)
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

        # Bellow we do filtering stuff
        captcha_array = []
        # loop our all detection boxes
        for i,b in enumerate(boxes[0]):
            for Symbol in range(NUM_CLASSES):
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
