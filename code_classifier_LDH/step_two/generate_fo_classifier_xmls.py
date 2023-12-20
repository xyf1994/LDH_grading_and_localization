import os
from tensorflow.keras.models import model_from_json
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import pickle
import cv2


tf.keras.backend.clear_session()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


ROOT_DIR = " "
LDH_DET_ROOT = " "
CLASSIFIER_ROOT_DIR = ""


iMAGING = " "

LDH_DET_IMGS_DIR =  " "
LDH_DET_PICKLE =  " "


best_LDH_weight = " "
best_LDH_path = " "
LDH_MODEL_WEIGHT = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_LDH_path,
        best_LDH_weight
    )
print(os.path.exists(LDH_MODEL_WEIGHT))
print(LDH_MODEL_WEIGHT, "exists")
print(os.path.getsize(LDH_MODEL_WEIGHT), "byte")

    # load model
TRAINED_JSON = os.path.join(
        CLASSIFIER_ROOT_DIR,
        best_LDH_path,
        " "
    )
    # Instantiate a model from JSON
json_file = open(TRAINED_JSON, 'r')
model_json = json_file.read()
json_file.close()

LDH_model = model_from_json(model_json)

LDH_model.load_weights(LDH_MODEL_WEIGHT) # Sets the state of the model.


print(os.path.exists(LDH_DET_PICKLE))
print(LDH_DET_PICKLE, "exists")
print(os.path.getsize(LDH_DET_PICKLE), "byte")

print(os.path.exists(TRAINED_JSON))
print(TRAINED_JSON, "exists")
print(os.path.getsize(TRAINED_JSON), "byte")


with open(LDH_DET_PICKLE, 'rb') as f:
    roi_detection = pickle.load(f)


print("Num of ROI detections: ", len(roi_detection.keys()))

print("partial view of roi_detection:")
print(dict(list(roi_detection.items())[0:2]))



nb_class = 4  
# dimensions of our images.
img_width, img_height = 150, 150

prelabel_folder = os.path.join(LDH_DET_ROOT, LDH_DET_IMGS_DIR)

grading = [' ', ' ', ' ', ' ']

predicted_scores_list = [] 
predicted_labels_list = []  


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


count_img = 0
for file in os.listdir(prelabel_folder):
    print(str(count_img) + " process file: " + file)
    if (not file.endswith('jpg')
            and not file.endswith('png')
            and not file.endswith('JPG')):
        print("***[NOT IMAGE]*** ", file, " is not an image")
        continue
    count_img += 1

    img_orig = keras.preprocessing.image.load_img(os.path.join(prelabel_folder, file))
    img = keras.preprocessing.image.img_to_array(img_orig)

    h, w, _ = img.shape
    print("--> img shape: ", img.shape, " <---")

    xml = [
        "<annotation>\n",
        "\t<folder>"+LDH_DET_IMGS_DIR+"</folder>\n",
        "\t<filename>" + file + "</filename>\n",
        "\t<path>" + os.path.join(prelabel_folder, file) + "</path>\n",
        "\t<source>\n",
        "\t\t<database>Unknown</database>\n",
        "\t</source>\n",
        "\t<size>\n",
        "\t\t<width>"+str(w)+"</width>\n",
        "\t\t<height>"+str(h)+"</height>\n",
        "\t\t<depth>1</depth>\n",
        "\t</size>\n",
        "\t<segmented>0</segmented>\n",
    ]

    for k, v in roi_detection[
        os.path.join(LDH_DET_IMGS_DIR, file)
    ].items():
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

        x = 1 / 255.0 * cropped_img

        if v == 3:
            print("\nv is 3, flip", v)
            x = cv2.flip(x, 1)

        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])

        prediction = []
        predicted_class = -1
        
        prediction = LDH_model.predict(images)
        predicted_class = np.argmax(prediction[0])


        # 将预测得分和标签附加到列表中
        predicted_scores_list.append(prediction[0])
        predicted_labels_list.append(grading[predicted_class])        

        LDH = [
            "\t<LDHect>\n",
            "\t\t<name>" + grading[predicted_class] + "</name>\n",
            "\t\t<pose>Unspecified</pose>\n",
            "\t\t<truncated>0</truncated>\n",
            "\t\t<difficult>0</difficult>\n",
            "\t\t<bndbox>\n",
            "\t\t\t<xmin>" + str(xmin) + "</xmin>\n",
            "\t\t\t<ymin>" + str(ymin) + "</ymin>\n",
            "\t\t\t<xmax>" + str(xmax) + "</xmax>\n",
            "\t\t\t<ymax>" + str(ymax) + "</ymax>\n",
            "\t\t</bndbox>\n",
            "\t</LDHect>\n",
        ]

        for e in LDH:
            xml.append(e)

    xml.append("</annotation>\n") # close off the xml tag

    with open(
        os.path.join(prelabel_folder, file.replace(iMAGING, "xml")),
        "w"
    ) as f:
        for e in xml:
            f.write(e)

num_files_combined = len(os.listdir(prelabel_folder))

print("num_files_combined in ", prelabel_folder, " = ", num_files_combined)

files_xml = [f for f in os.listdir(prelabel_folder) if "xml" in f]
print("num of xmls: ", len(files_xml))



# Calculation and drawing

import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from scipy.stats import norm
import numpy as np

labels_file_path = "  "

with open(labels_file_path, 'r') as file:
    lines = file.readlines()

for line in lines:
    parts = line.strip().split()
file_names, true_labels = zip(*[parts for parts in [line.strip().split() for line in lines] if len(parts) == 2])



predicted_labels = [grading.index(label) + 1 for label in predicted_labels_list]
true_labels = np.array(list(map(int, true_labels)))


def calculate_gwets_kappa(true_labels, predicted_labels):
    categories = np.unique(np.concatenate([true_labels, predicted_labels]))
    num_categories = len(categories)

    total_observed_agreement = np.sum(true_labels == predicted_labels)
    po = total_observed_agreement / len(true_labels)

    pe = np.sum(np.fromiter(
        ((np.sum(true_labels == c) / len(true_labels)) * (np.sum(predicted_labels == c) / len(true_labels))
         for c in categories),
        dtype=float
    ))

    ac1 = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0

    kappa_variance = (ac1 ** 2) / len(true_labels)

    z_score = ac1 / np.sqrt(kappa_variance)

    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    max_kappa = min(ac1 + 1.96 * np.sqrt(kappa_variance), 1.0)
    min_kappa = max(ac1 - 1.96 * np.sqrt(kappa_variance), -1.0)

    return ac1, min_kappa, max_kappa, p_value

gwets_kappa, min_kappa, max_kappa, p_value = calculate_gwets_kappa(true_labels, predicted_labels)

print("Gwet's Kappa:", gwets_kappa)
print("Gwet's Kappa Min:", min_kappa)
print("Gwet's Kappa Max:", max_kappa)
print("Gwet's Kappa p-value:", p_value)


conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=np.arange(1, 5))

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = grading
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)


for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(conf_matrix[i, j]), ha='center', va='center', color='black')

plt.xlabel('Predicted')
plt.ylabel('True')


accuracy_per_class = []
sensitivity_per_class = []
specificity_per_class = []

for i in range(len(classes)):
    true_positives = conf_matrix[i, i]
    false_positives = np.sum(conf_matrix[:, i]) - true_positives
    false_negatives = np.sum(conf_matrix[i, :]) - true_positives
    true_negatives = np.sum(conf_matrix) - (true_positives + false_positives + false_negatives)

    accuracy = (true_positives + true_negatives) / np.sum(conf_matrix)
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else 0.0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else 0.0

    accuracy_per_class.append(accuracy)
    sensitivity_per_class.append(sensitivity)
    specificity_per_class.append(specificity)

for i, grade in enumerate(grading):
    print(f"\nMetrics for {grade}:")
    print(f"{grade} Accuracy:", accuracy_per_class[i])
    print(f"{grade} Sensitivity:", sensitivity_per_class[i])
    print(f"{grade} Specificity:", specificity_per_class[i])



y_true = label_binarize(true_labels, classes=np.arange(1, 5))
y_scores = np.array(predicted_scores_list)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(4):
    fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


def calculate_metrics(true_labels, predicted_labels):
    metrics = {}

    metrics['overall_accuracy'] = accuracy_score(true_labels, predicted_labels)

    precision, recall, f1, support = precision_recall_fscore_support(true_labels, predicted_labels, average=None)

    for i, grade in enumerate(grading):
        metrics[f'{grade}_precision'] = precision[i]
        metrics[f'{grade}_recall'] = recall[i]
        metrics[f'{grade}_f1_score'] = f1[i]
        metrics[f'{grade}_support'] = support[i]

    return metrics

# Calculate ICC
def calculate_icc(true_labels, predicted_labels):
    categories = np.unique(np.concatenate([true_labels, predicted_labels]))
    num_categories = len(categories)

    # Create a matrix for observed and expected agreements
    observed_agreements = np.zeros((num_categories, num_categories))
    expected_agreements = np.zeros((num_categories, num_categories))

    for i in range(len(true_labels)):
        observed_agreements[true_labels[i] - 1, predicted_labels[i] - 1] += 1

    total_observed_agreements = np.sum(observed_agreements)

    for i in range(num_categories):
        for j in range(num_categories):
            expected_agreements[i, j] = (np.sum(observed_agreements[i, :]) * np.sum(observed_agreements[:, j])) / total_observed_agreements

    expected_agreement = np.sum(np.diagonal(expected_agreements))
    observed_agreement = np.sum(np.diagonal(observed_agreements))

    icc = (observed_agreement - expected_agreement) / (total_observed_agreements - expected_agreement)

    return icc

# Calculate metrics
metrics = calculate_metrics(true_labels, predicted_labels)

print("\nOverall Metrics:")
print("Overall Accuracy:", metrics['overall_accuracy'])

for grade in grading:
    print(f"\nMetrics for {grade}:")
    print(f"{grade} Precision:", metrics[f'{grade}_precision'])
    print(f"{grade} Recall:", metrics[f'{grade}_recall'])
    print(f"{grade} F1 Score:", metrics[f'{grade}_f1_score'])
    print(f"{grade} Support:", metrics[f'{grade}_support'])

icc = calculate_icc(true_labels, predicted_labels)
print("\nIntraclass Correlation Coefficient (ICC):", icc)



plt.subplot(1, 2, 2)
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
for i, color in zip(range(4), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'Class {grading[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

plt.show()



print("True Labels Shape:", true_labels.shape)
print("Predicted Labels Shape:", np.array(predicted_labels).shape)




import pickle
from datetime import datetime

current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f'D:/metrics_data_{current_datetime}.pickle'

metrics_data = {
    'conf_matrix': conf_matrix,
    'accuracy_per_class': accuracy_per_class,
    'sensitivity_per_class': sensitivity_per_class,
    'specificity_per_class': specificity_per_class,
    'roc_auc': roc_auc,
    'fpr': fpr,
    'tpr': tpr,
    'true_labels': true_labels,
    'predicted_labels': predicted_labels
}

with open(file_name, 'wb') as f:
    pickle.dump(metrics_data, f)

with open(file_name, 'rb') as f:
    loaded_metrics_data = pickle.load(f)

print(loaded_metrics_data)
print(true_labels)


def calculate_gwets_kappa_binary(true_labels, predicted_labels):
    true_labels_binary = np.where(np.isin(true_labels, [1, 2]), 0, 1)
    predicted_labels_binary = np.where(np.isin(predicted_labels, [1, 2]), 0, 1)

    categories = np.unique(np.concatenate([true_labels_binary, predicted_labels_binary]))
    num_categories = len(categories)


    total_observed_agreement = np.sum(true_labels_binary == predicted_labels_binary)
    po = total_observed_agreement / len(true_labels_binary)


    pe = np.sum(np.fromiter(
        ((np.sum(true_labels_binary == c) / len(true_labels_binary)) * (np.sum(predicted_labels_binary == c) / len(true_labels_binary))
         for c in categories),
        dtype=float
    ))


    ac1 = (po - pe) / (1 - pe) if (1 - pe) != 0 else 0.0


    kappa_variance = (ac1 ** 2) / len(true_labels_binary)

    z_score = ac1 / np.sqrt(kappa_variance)

    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))

    max_kappa = min(ac1 + 1.96 * np.sqrt(kappa_variance), 1.0)
    min_kappa = max(ac1 - 1.96 * np.sqrt(kappa_variance), -1.0)

    return ac1, min_kappa, max_kappa, p_value

gwets_kappa_binary, min_kappa_binary, max_kappa_binary, p_value_binary = calculate_gwets_kappa_binary(true_labels, predicted_labels)

print(f"Gwet's Kappa for binary classification: {gwets_kappa_binary:.4f}")
print(f"Minimum Kappa: {min_kappa_binary:.4f}")
print(f"Maximum Kappa: {max_kappa_binary:.4f}")
print(f"P-value: {p_value_binary:.4f}")



