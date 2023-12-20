import os
from tensorflow.keras.models import model_from_json
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pickle
import cv2
from heatmap_IG_utils import main_ig


print("TF version: ", tf.__version__)
print("cv2 version: ", cv2.__version__)

Root = " "


Classification = os.path.join(Root, " ")


Saved = " "  # no "/" at the end!
OBJ_DET_IMGS_DIR = " "
OBJ_DET_PICKLE = os.path.join(
            Root,
            OBJ_DET_IMGS_DIR,
            " "
        )


best__weight = " "
best__path = " "
weight = os.path.join(
        Classification,
        best__path,
        best__weight
    )
print(os.path.exists(weight))
print(weight, "exists")
print(os.path.getsize(weight), "byte")

    # load model
TRAINED_JSON = os.path.join(
        Classification,
        best__path,
        " "
    )
    # Instantiate a model from JSON
json_file = open(TRAINED_JSON, 'r')
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)

model.load_weights(weight) 


with open(OBJ_DET_PICKLE, 'rb') as f:
    roi_detection = pickle.load(f)


print("Num of ROI detections: ", len(roi_detection.keys()))

print("partial view of roi_detection:")
print(dict(list(roi_detection.items())[0:2]))


nb_class = 4
# dimensions of our images.
img_width, img_height = 150, 150
img_size = (img_width, img_height)

prelabel_folder = os.path.join(Root, OBJ_DET_IMGS_DIR)

grading = np.array([' ', ' ', ' ', ' '])

center_matrix = np.zeros(nb_class)
lateral_matrix = np.zeros(nb_class)
matrix = np.zeros(nb_class)

def scale_crop(xmin, xmax, ymin, ymax, factor, img_shape):

    cropped_w = xmax - xmin
    cropped_h = ymax - ymin
    xmin -= ((cropped_w * factor) // 2)
    ymin -= ((cropped_h * factor) // 2)

    xmax += ((cropped_w * factor) // 2)
    ymax += ((cropped_h * factor) // 2)

    height = img_shape[0]
    width = img_shape[1]

    return (
            int(max(xmin, 0)),
            int(min(xmax, width)),
            int(max(ymin, 0)),
            int(min(ymax, height))
            )

count_img = 0
for file in os.listdir(prelabel_folder):
    print("\n" + str(count_img) + " process file: " + file)
    if (not file.endswith('JPG')
            and not file.endswith('png')
            and not file.endswith('jpg')):
        print("***[NOT IMAGE]*** ", file, " is not an image")
        continue
    count_img += 1


    img_path = os.path.join(prelabel_folder, file)


    img_orig = keras.preprocessing.image.load_img(img_path)

    img = keras.preprocessing.image.img_to_array(img_orig)


    h, w, _ = img.shape
    detection_items = roi_detection[
        os.path.join(
            Root,
            OBJ_DET_IMGS_DIR,
            file
        )
    ].items()


    detection_items = sorted(
            detection_items,
            key=lambda item: item[0][0]
        )

    for (count, (k, v)) in enumerate(detection_items):
        ymin, xmin, ymax, xmax = k
        (xmin, xmax, ymin, ymax) = (
            int(xmin * w),
            int(xmax * w),
            int(ymin * h),
            int(ymax * h),
        )

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
        print("-cv2 resize: ", cropped_img.shape, cropped_img.dtype, cropped_img[0][0])

        x = 1 / 255.0 * cropped_img
        print("-normalize: ", x.shape, x.dtype, x[0][0])

        if v == 3:
            # print("\nv is 3, flip", v)
            x = cv2.flip(x, 1)

        # digress to a Tensor object for IG workflow
        img_tensor = tf.image.convert_image_dtype(x, tf.float32)
        # print("-convert_image_dtype: ", img_tensor.shape, img_tensor.dtype, img_tensor[0][0])

        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        # print("images (batch) shape: ", images.shape)

        prediction = []
        predicted_class = -1
        meta = {}
        meta["file_name"] = file
        meta["v"] = v
        # keep track of the  ROI from top to down
        meta["position_index"] = (count + 1)
        # save dir for jpegs, no "/" at the end!
        meta["Saved"] = os.path.join(Root, Saved)
        if v == 1:  
            prediction = model.predict(images)
            predicted_class = np.argmax(prediction[0])
            matrix[predicted_class] += 1
            main_ig(
                model,
                img_tensor,
                predicted_class,
                prediction,
                meta)

        else:
            continue

        print("\tprediction: ", prediction)
        print("\tpredicted_class: ", predicted_class, grading[predicted_class])

print("\n\n === IG generation finished ====")
print("matrix: ", matrix)
