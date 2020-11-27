import os
import math
import keras
from keras.layers import GlobalMaxPooling2D, Dense
from keras.models import Model
from keras_applications import inception_resnet_v2, resnet
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from config import config
from pretreatment import data_generator, test_generator

cfg = config()


def get_model(input_shape, classes, model_type):
    """
    :param input_shape: input_shape(height, width, channel)
    :param classes: classes
    :param model_type: 'ResNet50', 'ResNet101', 'ResNet152' or 'InceptionResNetV2'
    :return: model without compiling
    """
    if model_type == 'ResNet50':
        base_model = resnet.ResNet50(include_top=False,
                                     weights='imagenet',
                                     input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                                     classes=17,
                                     backend=keras.backend,
                                     layers=keras.layers,
                                     models=keras.models,
                                     utils=keras.utils)
    elif model_type == 'ResNet101':
        base_model = resnet.ResNet101(include_top=False,
                                      weights='imagenet',
                                      input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                                      classes=17,
                                      backend=keras.backend,
                                      layers=keras.layers,
                                      models=keras.models,
                                      utils=keras.utils)
    elif model_type == 'ResNet152':
        base_model = resnet.ResNet152(include_top=False,
                                      weights='imagenet',
                                      input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                                      classes=17,
                                      backend=keras.backend,
                                      layers=keras.layers,
                                      models=keras.models,
                                      utils=keras.utils)
    elif model_type == 'InceptionResNetV2':
        base_model = inception_resnet_v2.InceptionResNetV2(include_top=False,
                                                           weights='imagenet',
                                                           input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                                                           classes=17,
                                                           backend=keras.backend,
                                                           layers=keras.layers,
                                                           models=keras.models,
                                                           utils=keras.utils)
    else:
        raise ValueError('Model do not exist!')
    base_model.trainable = False
    x = base_model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(2048, activation='relu', kernel_regularizer=l2(0.0003))(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model


def get_total_num(image_path):
    files = os.listdir(image_path)
    return len(files)


def fit_model(model):
    check_point = ModelCheckpoint(cfg.best_model, monitor='val_accuracy', save_best_only=True, mode='max')
    model.fit_generator(generator=data_generator(cfg.train_dir, cfg.train_list, cfg.batch_size, 'train'),
                        steps_per_epoch=get_total_num(cfg.train_dir) / cfg.batch_size,
                        validation_data=data_generator(cfg.validation_dir, cfg.validation_list, cfg.batch_size, 'val'),
                        validation_steps=get_total_num(cfg.validation_dir) / cfg.batch_size,
                        epochs=cfg.epochs, verbose=1,
                        callbacks=[check_point])
    return model


def predict_model(model):
    predict = model.predict_generator(generator=test_generator(cfg.test_dir, cfg.batch_size),
                                      steps=math.ceil(get_total_num(cfg.test_dir) / cfg.batch_size),
                                      verbose=1)
    return predict
