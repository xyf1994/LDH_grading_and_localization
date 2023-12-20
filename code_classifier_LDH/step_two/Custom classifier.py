import tensorflow as tf
import numpy as np
import itertools
import os
from tensorflow.keras import backend as K

import datetime

print("TF version: ", tf.version.VERSION)

tf.keras.backend.clear_session()


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


img_width, img_height = 150, 150


train_data_dir = ' '
validation_data_dir = ' '

nb_classes = 4
nb_train_samples = 0
nb_validation_samples = 0
nb_sample_per_class = []
nb_val_sample_per_class = []

folders = ['1', '2', '3', '4']
for folder in folders:
    num_tr = len(os.listdir(os.path.join(train_data_dir, folder)))
    nb_train_samples += num_tr
    nb_sample_per_class.append(num_tr)

for folder in folders:
    num_val = len(os.listdir(os.path.join(validation_data_dir, folder)))
    nb_validation_samples += num_val
    nb_val_sample_per_class.append(num_val)

print("\nnb_train_samples: ", nb_train_samples)
print("\nnb_validation_samples: ", nb_validation_samples)
print("\nnb_sample_per_class: ", nb_sample_per_class)
print("\nnb_val_sample_per_class: ", nb_val_sample_per_class)
print("--")

epochs = 500
batch_size = 64

from functools import partial, update_wrapper

def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in itertools.product(range(nb_cl), range(nb_cl)):
        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

w_array = np.ones((4,4))
w_array[0, 1] = 1
w_array[0, 2] = 1
w_array[0, 3] = 1
w_array[1, 0] = float(nb_sample_per_class[0])/float(nb_sample_per_class[1])
w_array[1, 2] = float(nb_sample_per_class[0])/float(nb_sample_per_class[1])
w_array[1, 3] = float(nb_sample_per_class[0])/float(nb_sample_per_class[1])
w_array[2, 0] = float(nb_sample_per_class[0])/float(nb_sample_per_class[2])
w_array[2, 1] = float(nb_sample_per_class[0])/float(nb_sample_per_class[2])
w_array[2, 3] = float(nb_sample_per_class[0])/float(nb_sample_per_class[2])
w_array[3, 0] = float(nb_sample_per_class[0])/float(nb_sample_per_class[3])
w_array[3, 1] = float(nb_sample_per_class[0])/float(nb_sample_per_class[3])
w_array[3, 2] = float(nb_sample_per_class[0])/float(nb_sample_per_class[3])


ncce = partial(w_categorical_crossentropy, weights=w_array)
ncce.__name__ = 'w_categorical_crossentropy'

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model

def resnet50_classifier_model(input_shape, num_classes):

    input_tensor = Input(shape=input_shape)

    # Stage 0
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Stage 1
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Stage 2
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 3
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 4
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')


    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.2)(x)
    
    x = Dense(num_classes, activation='softmax', name='output')(x)


    model = Model(inputs=input_tensor, outputs=x, name='resnet50')

    return model


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """Identity block"""
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    x = x + input_tensor  # Skip connection
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """Convolutional block"""
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides, name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1')(shortcut)

    x = x + shortcut  # Skip connection
    x = Activation('relu')(x)
    return x



input_shape = (img_height, img_width, 3)  
num_classes = nb_classes  
model = resnet50_classifier_model(input_shape, num_classes)


model.compile(loss=ncce,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()



train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=5,
    )

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# iterator that returns image_batch, label_batch pairs
for image_batch, label_batch in train_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break
for image_batch, label_batch in validation_generator:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    break


formatted_time = datetime.datetime.now().strftime("%m%d-%H%M")
save_model_dir = "resnet50_Dropout_Date{}".format(formatted_time)

if not os.path.exists(save_model_dir):
    print(save_model_dir, " will be created")
    os.makedirs(save_model_dir)

store_model_json_name = "{}.json".format(formatted_time)
# store model
model_json = model.to_json()
model_json_path = os.path.join(save_model_dir, store_model_json_name)
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

checkpoint_filepath = os.path.join(
    save_model_dir,
    "{0}_Ep{{epoch:02d}}_ValAcc{{val_accuracy:.3f}}_ValLoss{{val_loss:.2f}}.h5"
    .format(save_model_dir)
)

callbacks = [
   
    tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        save_best_only=False,
        save_weights_only=True,
        verbose=1,
        save_freq="epoch"),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5, # usually 0.1
        patience=10,
        verbose=1),
    tf.keras.callbacks.CSVLogger(
        filename=os.path.join(
            save_model_dir,
            '{}.csv'.format(save_model_dir)
        ),
        append=False,
        separator=','),
]


train_steps = np.ceil(nb_train_samples / batch_size)
print("len train_generator: ", len(train_generator))

val_steps = np.ceil(nb_validation_samples / batch_size)
print("len validation_generator: ", len(validation_generator))

train_history = model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=val_steps,
    callbacks=callbacks)
