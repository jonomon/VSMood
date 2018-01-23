import numpy as np
import pandas as pd
import random
import logging
import argparse
import os
import matplotlib.pyplot as plt

random.seed(1114)
np.random.seed(129)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.metrics import balanced_accuracy_score

from vsbrnn.run_vsb_sequence import run_vsb_sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## For basic usage check python main.py -h

def main():
    data_type_options = ["fix", "glance", "fix-sequence"]
    multi_instance_options = ['mean', '2d-mean', 'max-likelihood', 'similar', 'log-prob']
    cnn_layers_options = ['1', '2', 'none']
    region_model_type_options = ["semantic5", "grid9", "semantic8", "grid16"]
    use_vsb_options = ['scan_path', 'glance_dur']
    use_img_options = ['img_type', 'img_pos']
    
    logging.basicConfig(filename='log.txt', level=logging.DEBUG, filemode="w")
    parser = argparse.ArgumentParser(description='Run RNN for bipolar.')
    parser.add_argument('data_type', type=str, help='options: {}'.format(data_type_options))
    parser.add_argument('states', type=int, help='states')
    parser.add_argument('multi_instance', type=str,
                        help='Multi instance options {}'.format(multi_instance_options))
    parser.add_argument('--region_model_type', type=str,
                        help='region model types {} default region_clinical'.format(
                            region_model_type_options))
    parser.add_argument('--cnn_layers', type=str,
                        help='cnn options {}'.format(cnn_layers_options))
    parser.add_argument('--max_len', type=int, help='max length of sequence')
    parser.add_argument('--use_vsb', type=str, nargs="+",
                        help='VSB features only with glance data_type, options: {}'.format(
                            use_vsb_options))
    parser.add_argument('--use_img', type=str, nargs="+",
                        help='should use image properties, options: {}'.format(use_img_options))
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--print_sub', dest='print_sub', action='store_true')
    parser.add_argument('--plot', dest='plot', action='store_true')

    args = parser.parse_args()
    data_type = args.data_type
    states = args.states
    max_len = args.max_len
    use_vsb = args.use_vsb
    use_img = args.use_img
    region_model_type = args.region_model_type
    cnn_layers = args.cnn_layers

    multi_instance = args.multi_instance
    verbose = args.verbose
    print_sub = args.print_sub
    plot = args.plot

    logging.debug("Running %s with states=%s, mi=%s, max_length=%s, use_vsb=%s, use_img=%s",
                  data_type, states, multi_instance, max_len, use_vsb, use_img)
    print("Running {} with states={}, mi={}, max_length={}, use_vsb={}, use_img={} region_model={} cnn_layers={}".format(data_type, states, multi_instance, max_len, use_vsb, use_img,
        region_model_type, cnn_layers))

    if data_type not in data_type_options:
        print("{} not an available data_type option".format(data_type))
        return
    if data_type == "fix" and use_vsb:
        print("VSB parameters are not available when in fixation")
        return
    if multi_instance not in multi_instance_options:
        print("{} not available option for multi_instance".format(multi_instance))
        return
    if region_model_type != None and region_model_type not in region_model_type_options:
        print("{} not available option for region_model_type".format(region_model_type))
        return

    if cnn_layers != None and cnn_layers not in cnn_layers_options:
        print("{} not available option for cnn_layers".format(cnn_layers))
        return    

    sub_cat, sub_prob = run_vsb_sequence(data_type, states, max_len,
                                         use_vsb, use_img, region_model_type,
                                         cnn_layers, multi_instance,
                                         verbose=verbose, print_sub=print_sub)
    sub_prob = np.array(sub_prob)
    df = pd.DataFrame({"cat": sub_cat, "prob": sub_prob})
    df.to_csv("output/{}-{}-{}-{}.csv".format(data_type, states, region_model_type, cnn_layers))
    clf = LogisticRegression(class_weight="balanced")
    clf.fit(sub_prob.reshape(-1, 1), sub_cat)
    y_predicted = clf.predict(sub_prob.reshape(-1, 1))
    
    auc_val = roc_auc_score(sub_cat, sub_prob)
    acc_val = accuracy_score(sub_cat, y_predicted)
    b_acc_val = balanced_accuracy_score(sub_cat, y_predicted)
    log_loss_val = log_loss(sub_cat, sub_prob)

    print("Avg auc={} acc_val={} b_acc_val={} log_loss_val={}\n\n".format(
        auc_val, acc_val, b_acc_val, log_loss_val))

    if plot:
        from sklearn.metrics import roc_curve, auc
        plt.clf()
        fpr, tpr, _ = roc_curve(sub_cat, sub_prob, pos_label=1)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, color='#FF69B4',
                 label='ROC curve (area = {})'.format(round(auc_val, 2)))
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        filename = "img/{} with states={}, mi={}".format(data_type, states, multi_instance)
        plt.savefig(filename + ".png")
        pd.DataFrame([sub_cat, sub_prob]).T.to_csv(filename + ".csv")

if __name__ == "__main__":
    main()


