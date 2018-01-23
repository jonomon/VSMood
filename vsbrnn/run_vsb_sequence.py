import numpy as np
import pandas as pd

from vsbrnn.data.data_importer import DataImporter
from vsbrnn.data.data_creator import DataCreator
from vsbrnn.training import RnnTrain, RNNFeatureTrain
from vsbrnn.multi_instance import MultiInstance

from sklearn.model_selection import train_test_split, StratifiedKFold

import matplotlib.pyplot as plt
from vsbrnn.utils import makeGaussian
from data.region_model import RegionModel, FaceRegionModel_grid9, FaceRegionModel_semantic5
from data.region_model import FaceRegionModel_grid16, FaceRegionModel_semantic8
from data.region_model import FaceRegionModel4

## GAUSSIAN SIGMA=6 from 5

def run_vsb_sequence(data_type, states, max_len, use_vsb,
                     use_img, region_model_type, cnn_layer, multi_instance,
                     verbose, print_sub):    
    di = DataImporter()
    if region_model_type == "semantic5":
        region_model = FaceRegionModel_semantic5()
    elif region_model_type == "grid9":
        region_model = FaceRegionModel_grid9()
    elif region_model_type == "semantic8":
        region_model = FaceRegionModel_semantic8()
    elif region_model_type == "grid16":
        region_model = FaceRegionModel_grid16()
    else:
        region_model = FaceRegionModel4()
    dc = DataCreator(di, data_type, max_len=max_len, use_vsb=use_vsb,
                     model=region_model,
                     use_img=use_img)
    dc.filter_cats(["BD", "D"])
    subject_list = dc.get_unique_subject_list()[::-1]
    cat_list = dc.get_cat_of_subjects(subject_list)
    sub_cat = []
    sub_prob = []
    for idx, test_sub in enumerate(subject_list):
        non_test_cats = np.delete(cat_list, idx)
        non_test_subs = np.delete(subject_list, idx)

        X_test, y_test = dc.get_data_for_subjects(test_sub)
        predicts = []
        skf = StratifiedKFold(n_splits=3)
        for train_index, valid_index in skf.split(non_test_subs, non_test_cats):
            # train_subs, valid_subs, train_cats, valid_cats = train_test_split(
            #                 non_test_subs, non_test_cats, test_size=0.33,
            #                 stratify=non_test_cats, random_state=i)
            train_subs = non_test_subs[train_index]
            valid_subs = non_test_subs[valid_index]
            if data_type == "fix-sequence":
                X_train, y_train = dc.get_data_for_subjects(train_subs)
                X_valid, y_valid = dc.get_data_for_subjects(valid_subs)
                trainer = RNNFeatureTrain(cnn_layer, states=states, verbose=verbose)
                trainer.do_training(X_train, y_train,
                                    X_valid, y_valid)
                preds = trainer.predict(X_test)                
            else:
                X_train, y_train = dc.get_data_for_subjects(train_subs)
                X_valid, y_valid = dc.get_data_for_subjects(valid_subs)
                trainer = RnnTrain(states=states, verbose=verbose)
                trainer.do_training(X_train, y_train, X_valid, y_valid)
                preds = trainer.predict(X_test)

            mi = MultiInstance(multi_instance, X_train, y_train, X_test, trainer)
            mi_preds = mi.get_pred(preds)
            predicts.append(mi_preds)
        mean_test_predict = np.mean(predicts)
        sub_prob.append(mean_test_predict)
        sub_cat.append(y_test[0, 1])
        log = "Sub = {}, Mean test = {} cat = {}".format(
             test_sub, mean_test_predict, y_test[0])
        if print_sub or verbose:
            print(log)
    return sub_cat, sub_prob
