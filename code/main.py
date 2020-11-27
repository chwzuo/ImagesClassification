from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.optimizers import Adam
from config import config
from cnn_model import get_model, fit_model, predict_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 3, 4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def save2csv(predicted):
    p = []
    file_list = os.listdir(cfg.test_dir)
    file_list.sort()
    for values in predicted:
        str_values = ' '.join(str(value) for value in values)
        p.append(str_values)
    data = pd.DataFrame({'id': file_list, 'predicted': p})
    data.to_csv(cfg.save_csv, index=False)


if __name__ == '__main__':
    cfg = config()
    input_shape = [cfg.images_height, cfg.images_width, cfg.images_channel]
    if os.path.exists(cfg.best_model):
        model = load_model(cfg.best_model)
        if cfg.continue_fitting:
            model = fit_model(model)
    else:
        model = get_model(input_shape=input_shape, classes=cfg.n_classes, model_type=cfg.model)
        model.compile(optimizer=Adam(lr=cfg.learning_rate, beta_1=0.9, beta_2=0.99, epsilon=1e-08, decay=1e-6),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model = fit_model(model)
    predict = predict_model(model)
    predict = np.argsort(-predict, axis=1)[:, :3]
    print(predict)
    save2csv(predict)
