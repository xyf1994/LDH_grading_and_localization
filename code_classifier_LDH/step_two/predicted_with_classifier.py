import os
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

import tensorflow as tf
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import cv2

'''      jpg         改成        JPG'''
# import visualization_utils as vis_util
from utils import visualization_utils as vis_util


convert_to_number = {'normal':1, 'grade1': 2, 'grade2':3, 'grade3':4, '1':1, '2':2, '3':3, '4':4}


# scale the crop
def scale_crop(xmin, xmax, ymin, ymax, factor, img_shape):
    """
    the imgs top left corner is (0,0),
    x min-max is from point A,
    y min-max is from point B
    """
    cropped_w = xmax - xmin
    cropped_h = ymax - ymin
    xmin -= ((cropped_w * factor) // 2)
    ymin -= ((cropped_h * factor) // 2)

    xmax += ((cropped_w * factor) // 2)
    ymax += ((cropped_h * factor) // 2)

    # cv2 img shape information
    height = img_shape[0]
    width = img_shape[1]

    return (
            int(max(xmin, 0)),
            int(min(xmax, width)),
            int(max(ymin, 0)),
            int(min(ymax, height))
            )


# dimensions of our images.
# img_width, img_height = 84, 84
img_width, img_height = 150, 150


# load detected bounding box for axial and sagittal images
# for Resnet

with open('D:/models/research/.idea/sag_for_test/sag/sag.pickle', 'rb') as f:
    sag_detection = pickle.load(f)
    sag_detection_item_predix = "D:/models/research/.idea/sag_for_test/sag"

print("Num of sag detections: ", len(sag_detection.keys()))

print("partial view of sag_detection:")
print(dict(list(sag_detection.items())[0:2]))


# load classification models
CLASSIFIER_ROOT_DIR = "D:/models/research/.idea/Classification/"


# weights
best_sag_weight = "sag_resnetscale150V1_150x150bat128_6LDropout_Date1123-1701_Ep500_ValAcc0.839_ValLoss4.98.h5"# h5文件，在epoch中寻找最好的h5填进去
best_sag_path = "sag_resnetscale150V1_150x150bat128_6LDropout_Date1123-1701"
SAG_MODEL_WEIGHT = os.path.join(
    CLASSIFIER_ROOT_DIR,
    best_sag_path,
    best_sag_weight
)
print(os.path.exists(SAG_MODEL_WEIGHT))
print(SAG_MODEL_WEIGHT, "exists")
print(os.path.getsize(SAG_MODEL_WEIGHT), "byte")

# load model
TRAINED_JSON = os.path.join(
    CLASSIFIER_ROOT_DIR,
    best_sag_path,
    "sag_train1123-1701.json"
)
# Instantiate a model from JSON
json_file = open(TRAINED_JSON, 'r')
model_json = json_file.read()
json_file.close()

sag_model = model_from_json(model_json)

sag_model.load_weights(SAG_MODEL_WEIGHT) # Sets the state of the model.

print("Loaded sag_model from json and weights")


nb_class = 4

category_index = {1: {'id': 1, 'name': 'normal'}, 2: {'id': 2, 'name': 'grade1'}, 3: {'id': 3, 'name': 'grade2'}, 4: {'id': 4, 'name': 'grade3'}}


cwd = 'D:/models/research/.idea/sag_for_test'
for folder in os.listdir(cwd):
    print("process folder " + folder)
    
    for file in os.listdir(os.path.join(cwd, folder)):
        if '.xml' in file:
            print('process file ' + file)

            img_orig = image.load_img(os.path.join(cwd, folder, file.replace('xml', 'JPG')))
            img = image.img_to_array(img_orig)

            h, w, _ = img.shape

            data = ET.parse(os.path.join(cwd, folder, file))
            root = data.getroot()

            if 'axial' in folder:#   在cwd文件夹内加入axial文件夹

                
                print("\n##### axial-predict #####\n")



            elif 'sag' in folder:#    在cwd文件夹内加入sag文件夹

                boxes = []
                gradings = []
                for o in root.findall('object'):
                    gradings.append(convert_to_number[o.find('name').text])
                    xmin = int(o.find('bndbox').find('xmin').text)
                    xmax = int(o.find('bndbox').find('xmax').text)
                    ymin = int(o.find('bndbox').find('ymin').text)
                    ymax = int(o.find('bndbox').find('ymax').text)

                    boxes.append((xmin, xmax, ymin, ymax))

                # boxes = sorted(boxes, key=lambda x: x[2])

                visualize_test_box = np.zeros((len(boxes), 4))
                visualize_test_classes = np.zeros((len(boxes)), dtype=np.uint8)

                labels = []

                for i in range(0, len(boxes)):
                    visualize_test_box[i] = np.array([boxes[i][2], boxes[i][0], boxes[i][3], boxes[i][1]])
                    # visualize_test_classes[i] = int(sag_label[file][i])
                    # labels.append([boxes[i], int(sag_label[file][i])-1])

                    visualize_test_classes[i] = gradings[i]
                    labels.append([boxes[i], gradings[i]-1])

                test_img = np.copy(img_orig)

                vis_util.visualize_boxes_and_labels_on_image_array(
                    test_img,
                    visualize_test_box,
                    visualize_test_classes,
                    np.ones(visualize_test_classes.shape[0], dtype=np.uint8),
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=20,
                    line_thickness=4)


                cv2.imwrite(os.path.join(cwd, 'sag-test-with-prob', file.replace('xml', 'JPG')), test_img) #sag-test-with-prob可以更换名字，在cwd文件夹内


                nb_prediction = len(sag_detection[sag_detection_item_predix+'/'+file.replace('xml','JPG')].keys())
                visualize_predict_box = np.zeros((nb_prediction, 4))
                visualize_predict_classes = np.zeros(nb_prediction, dtype=np.uint8)
                visualize_predict_scores = np.zeros(nb_prediction)

                count_prediction = 0
                for k,v in sag_detection[sag_detection_item_predix+'/'+file.replace('xml','JPG')].items():
                    ymin, xmin, ymax, xmax = k
                    (xmin, xmax, ymin, ymax) = (int(xmin*w), int(xmax*w), int(ymin*h), int(ymax*h))
                    box = (xmin, xmax, ymin, ymax)

                    visualize_predict_box[count_prediction] = np.array([ymin, xmin, ymax, xmax])

                    print(">>>>before scaling, xmin, xmax, ymin, ymax: ", xmin, xmax, ymin, ymax)

                    scale_factor = 0.5 # 0.5 for scale by 150%
                    cv2_shape = [h, w]
                    (xmin_for_pred, xmax_for_pred, ymin_for_pred, ymax_for_pred) = scale_crop(
                        xmin, xmax, ymin, ymax, scale_factor, cv2_shape)

                    print("<<<<after scaling, xmin, xmax, ymin, ymax: ",
                        xmin_for_pred, xmax_for_pred, ymin_for_pred, ymax_for_pred)

                    cropped_img = img[ymin_for_pred:ymax_for_pred, xmin_for_pred:xmax_for_pred, :]
                    cropped_img = cv2.resize(
                        cropped_img,
                        (img_width, img_height),
                        # change this to inter_linear
                        interpolation=cv2.INTER_LINEAR
                    )

                    x = 1 / 255.0 * cropped_img
                    x = np.expand_dims(x, axis=0)

                    images = np.vstack([x])

                    prediction = sag_model.predict(images)
                    predicted_class = np.argmax(prediction[0])

                    visualize_predict_classes[count_prediction] = predicted_class+1
                    visualize_predict_scores[count_prediction] = prediction[0][predicted_class]

                    count_prediction += 1


                predict_img = np.copy(img_orig)
                vis_util.visualize_boxes_and_labels_on_image_array(
                    predict_img,
                    visualize_predict_box,
                    visualize_predict_classes,
                    visualize_predict_scores,
                    category_index,
                    use_normalized_coordinates=False,
                    max_boxes_to_draw=20,
                    min_score_thresh=.0,
                    line_thickness=1)

                cv2.imwrite(os.path.join(cwd, 'sag-predict', file.replace('xml', 'JPG')), predict_img) # sag-predict 可以更换名字，在cwd文件夹内

            else:
                continue