import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import argparse

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Initiate argument parser
parser = argparse.ArgumentParser(
    description="model inference sample")
parser.add_argument("-m",
                    "--saved_model_dir",
                    help="Path to saved model directory.",
                    type=str, default=" ")
parser.add_argument("-l",
                    "--labels_path",
                    help="Path to the labels (.pbtxt) file.", type=str, default=" ") 

parser.add_argument("-i",
                    "--images_dir",
                    help="Path of input images file.", type=str, default=" ")
parser.add_argument("-o",
                    "--output_inference_result",
                    help="Path of output inference result file.", type=str, default=' ')
args = parser.parse_args()


import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# PATH_TO_SAVED_MODEL      = "exported_models/my_model/saved_model"
# PATH_TO_LABELS           = "annotations/label_map.pbtxt"
# PATH_TO_IMAGES           = "images/test"
# PATH_TO_INFERENCE_RESULT = 'inference_result/'

PATH_TO_SAVED_MODEL      = args.saved_model_dir
PATH_TO_LABELS           = args.labels_path
PATH_TO_IMAGES           = args.images_dir
PATH_TO_INFERENCE_RESULT = args.output_inference_result

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import os
import pickle

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def load_images_path(images_dir):
    images_path_list = []

    images_filename_list =  os.listdir(images_dir)
    for img_path in images_filename_list:
        if img_path.endswith(".JPG") == True:
            img_path = os.path.join('%s/%s' % (images_dir, img_path))
            images_path_list.append(img_path)

    return images_path_list

IMAGE_PATHS = load_images_path(PATH_TO_IMAGES)


detection_result = {}

for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')

    image_np = load_image_into_numpy_array(image_path)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # 将标记框信息保存至字典中
    boxes = detections['detection_boxes']
    classes = detections['detection_classes']
    scores = detections['detection_scores']
    image_result = {}
    
    for box, cls, score in zip(boxes, classes, scores):
        if score > 0.3:
            xmin, ymin, xmax, ymax = box.tolist()
            # 
            #    在这里使用(xmin, ymin, xmax, ymax)的顺序，和后续的代码进行匹配
            #       
            #
            #     一般来说，输出坐标为(ymin, xmin, ymax, xmax)的顺序
            image_result[(xmin, ymin, xmax, ymax)] = category_index[cls]['name']
    detection_result[image_path] = image_result

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=200,
          min_score_thresh=.30,
          agnostic_mode=False)

    plt.figure()
    image_filename = os.path.join(PATH_TO_INFERENCE_RESULT, os.path.basename(image_path))
    plt.imsave(image_filename, image_np_with_detections)
    print('Done')


# 将字典保存至pickle文件中
with open(' ', 'wb') as f:
    pickle.dump(detection_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('pickle_saved')
    