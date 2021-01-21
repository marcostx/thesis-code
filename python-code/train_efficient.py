import os
import torch
import sys
import numpy as np
import glob
from collections import Counter

from collections import OrderedDict
from efficientnet.tfkeras import EfficientNetB3
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from tensorflow.keras.applications import ResNet50
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adadelta, RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from efficientnet.tfkeras import  preprocess_input
# from keras.callbacks import EarlyStopping, ModelCheckpoint
# from tensorflow.keras.models import Model
# from keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
import h5py
# from utils.args import get_args
import argparse
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def getArgs():
    argparser = argparse.ArgumentParser(description=__doc__)

    #argparser.add_argument('-dp','--datapath', default='/Users/marcostexeira/pay-attention-pytorch/data/raw_frames/violentflow', help='Directory containing data sequences', type=str)
    argparser.add_argument('-dp', '--datapath', default='/home/datasets/',
                           help='Directory containing data sequences', type=str)
    argparser.add_argument('-dn', '--dataset_name',
                           default='rwf', help='dataset name ', type=str)

    args = argparser.parse_args()
    return args

# def preprocessing_function(x):
#     x = _preprocess_input(x)
#     # x = ((x/255) - mean) / std
#     # x = (x / 255)
#     return x

def freeze_layers(model):
    for i in model.layers:
        i.trainable = False
        if isinstance(i, Model):
            freeze_layers(i)
    return model

def build_model_v2():
    # inp = Input((244,244,3))
    base_model = ResNet50(include_top = False, weights ='imagenet', pooling='avg', input_shape=(224,224,3))
    head_model = Dense(2, activation="softmax")(base_model.output)

    model = Model(inputs=base_model.input, outputs=head_model)
    #for layer in model.layers:
    #    if hasattr(layer, 'kernel_regularizer'):
    #        layer.kernel_regularizer = regularizers.l1_l2(0.3)

    # model = Model(inputs=base_model.input, outputs=head_model)

    # opt = RMSprop(lr=2e-6, decay=1e-9)
    opt = Adam(lr=1e-3)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model

def build_model():
    base_model = EfficientNetB3(
        weights='imagenet', drop_connect_rate=0.5,  include_top=False, input_shape=(300,300,3))

    pooling = GlobalMaxPooling2D(name="gap")(base_model.output)
    norm = BatchNormalization()(pooling)
    
    drop = Dropout(0.5)(norm) 
    head_model = Dense(2, activation="softmax")(drop)

    model = Model(inputs=base_model.input, outputs=head_model)
    # for layer in model.layers:
    #    if hasattr(layer, 'kernel_regularizer'):
    #        layer.kernel_regularizer = regularizers.l1_l2(0.2)

    # opt = RMSprop(lr=2e-6, decay=1e-9)
    opt = Adam(lr=1e-5)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


def train_model_v2(model, train_generator, test_generator, epochs=50, verbose=1):
    filepath = 'finetuned_mobilenet_rwf_checkpoint.h5'
    early_stopping = EarlyStopping(patience=5)
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    callbacks = [early_stopping, checkpoint]
    
    model.fit(train_generator,
              epochs=epochs,
              validation_data=test_generator,
              shuffle=True,
              callbacks=callbacks,
              verbose=1)
    return model

def train_model(model, train_generator, test_generator, epochs=15, verbose=1):
    filepath = 'finetuned_efficientnet_rwf_checkpoint_flow.h5'
    # early_stopping = EarlyStopping(patience=5)
    checkpoint = ModelCheckpoint(filepath, save_best_only=True)
    callbacks = [checkpoint]
    
    model.fit(train_generator,
              epochs=epochs,
              validation_data=test_generator,
              shuffle=True,
              callbacks=callbacks,
              verbose=1)
    return model


def test_model(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test, verbose=1)

    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def main():
    args = getArgs()

    np.random.seed(0)
    # Dataset path
    data_path = args.datapath
    datasetName = args.dataset_name
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    train_generator = train_datagen.flow_from_directory(  
        '/home/datasets/rwf-2000-frames-of/train',
        target_size=(300, 300),
        batch_size=16,
        color_mode="rgb",
        shuffle=True,
        class_mode='categorical')
    
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        '/home/datasets/rwf-2000-frames-of/val',
        target_size=(300, 300),
        batch_size=16,
        color_mode="rgb",
        class_mode='categorical')

    model = build_model()

    model = train_model(model, train_generator, test_generator)

    model = freeze_layers(model)
    model.save("finetuned_efficientnet_rwf_flow.h5")
    # model = load_model("finetuned_efficientnet_rwf.h5")
 
    # preds = model.predict_generator(test_generator, steps=2)
    # print(preds)


if __name__ == "__main__":
    main()
    print("Done!")