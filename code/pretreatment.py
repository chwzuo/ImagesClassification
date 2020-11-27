import os
import numpy as np
from PIL import Image
from random import shuffle
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from config import config

# load configuration
cfg = config()


def __get_image_label(image_path, list_path):
    """
    :param image_path: training/validation images directory
    :param list_path: training/validation images list (subset)
    :return: images, labels
    """
    image = []
    label = []
    for lst in list_path:
        lst = lst.rstrip()
        img = Image.open(image_path + lst).resize((cfg.images_width, cfg.images_height))
        img = np.array(img)
        img = img / 127.5 - 1
        image.append(img)
        label.append(lst.split('_')[0])
    image = np.array(image)
    label = np.array(label)
    return image, to_categorical(label, cfg.n_classes)


def __get_image(image_path, list_path):
    """
    :param image_path: training/validation images directory
    :param list_path: training/validation images list (subset)
    :return: images
    """
    image = []
    for lst in list_path:
        lst = lst.rstrip()
        img = Image.open(image_path + lst).resize((cfg.images_width, cfg.images_height))
        img = np.array(img)
        img = img / 127.5 - 1
        image.append(img)
    image = np.array(image)
    return image


def data_generator(image_path, list_path, batch_size, type):
    """
    :param image_path: training/validation images directory
    :param list_path: training/validation images list
    :param batch_size: batch_size
    :return: images, label
    """
    while True:
        with open(list_path, 'r') as files:
            files = list(files)
            shuffle(files)
            for i in range(0, len(files), batch_size):
                end = min(len(files), i + batch_size)
                image, label = __get_image_label(image_path, files[i: end])
                if cfg.generator and type == 'train':
                    images, labels = [], []
                    data_gen = ImageDataGenerator(#featurewise_center=True, featurewise_std_normalization=True,
                                                  rotation_range=30,
                                                  width_shift_range=0.2, height_shift_range=0.2,
                                                  horizontal_flip=True)
                    data_gen.fit(image)
                    iterator = data_gen.flow(image, label, batch_size=cfg.batch_size)
                    i = 0
                    for image, label in iterator:
                        images.extend(image)
                        labels.extend(label)
                        i += 1
                        if i == cfg.gen_num: break
                    image, label = np.array(images), np.array(labels)
                yield image, label


def test_generator(image_path, batch_size):
    """
    :param image_path: testing images directory
    :param batch_size: batch size
    :return: images
    """
    while True:
        files = os.listdir(image_path)
        files.sort()
        for i in range(0, len(files), batch_size):
            end = min(len(files), i + batch_size)
            image = __get_image(image_path, files[i: end])
            yield image
