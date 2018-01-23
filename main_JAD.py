import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc

from vsbrnn.data.data_creator import DataCreator
from vsbrnn.data.data_importer import DataImporter
from vsbrnn.training import RnnTrain
from vsbrnn.utils import get_log_likelihood
from vsbrnn.data.region_model import FaceRegionModel4

np.random.seed(616)
#python main_psy_paper.py 32 13 --plot true

def plotROC(cats, preds, xlabel=None, filename=None):
    fpr, tpr, _ = roc_curve(cats, preds, pos_label=1)
    auc_val = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='#FF69B4', label='ROC curve (area = {})'.format(round(auc_val, 2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    if xlabel:
        plt.xlabel('False Positive Rate\n' + xlabel)
    else:
        plt.xlabel('False Positive Rate')
        
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig("img/" + filename + ".png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run seq2seq for apathy.')
    parser.add_argument('states', type=int, help="LSTM states")
    parser.add_argument('epochs', type=int, help="iterations")
    parser.add_argument('--plot', type=bool)
    
    args = parser.parse_args()
    states = args.states
    epochs = args.epochs
    plot = args.plot if args.plot is not None else False

    print("Running states={} epochs={}".format(states, epochs))
    # Get data
    di = DataImporter()
    bd_d_dc = DataCreator(di, "fix", max_len=None, model=FaceRegionModel4(),
                          use_vsb=None, use_img=["img_type"])
    bd_d_dc.filter_cats(["BD", "D"])
    other_dc = DataCreator(di, "fix", max_len=None, model=FaceRegionModel4(),
                           use_vsb=None, use_img=["img_type"])
    other_dc.filter_cats(["BR", "R", "C"])

    # Hold out data
    subject_list = bd_d_dc.get_unique_subject_list()
    cat_list = bd_d_dc.get_cat_of_subjects(subject_list)

    train_subs, hold_out_subs, train_cats, hold_out_cat = train_test_split(
        subject_list, cat_list,
        test_size=0.33, stratify=cat_list, random_state=23422)
    # Results 1) Leave one out classification
    mean_preds = []
    mean_cats = []
    classification = []
    for idx, test_sub in enumerate(train_subs):
        llo_train_subs = np.delete(train_subs, idx)
        X_train, y_train = bd_d_dc.get_data_for_subjects(llo_train_subs)
        X_test, y_test = bd_d_dc.get_data_for_subjects(test_sub)
        trainer = RnnTrain(states=states, verbose=False)
        trainer.do_simple_fix_training(X_train, y_train, epochs=epochs)
        X_predicts = trainer.predict(X_train).reshape(-1, 1)
        n_d, bins_d, _ = plt.hist(
            X_predicts[y_train[:, 1]==1], facecolor='green', alpha=0.5)
        n_bd, bins_bd, _ = plt.hist(
            X_predicts[y_train[:, 1]==0], facecolor='red', alpha=0.5)

        preds = trainer.predict(X_test)
        log_like = np.mean([get_log_likelihood(a, n_bd, bins_bd, n_d, bins_d) for a in preds])
        mean_preds.append(log_like)
        mean_cats.append(y_test[0, 1])
        classification.append(np.mean(log_like)>0)
        print("Training for {} log_like {}".format(test_sub, log_like))
    llo_auc = roc_auc_score(mean_cats, mean_preds)
    print("LLO AUC score = {}".format(llo_auc))
    classification = np.array(classification)
    mean_cats = np.array(mean_cats)
    cm = confusion_matrix(mean_cats, classification)
    print("CM = {}".format(cm))

    # Results 2) Predict remitted and controls
    X_train, y_train = bd_d_dc.get_data_for_subjects(train_subs)
    trainer2 = RnnTrain(states=states, verbose=False)
    trainer2.do_simple_fix_training(X_train, y_train, epochs=epochs)
    X_predicts = trainer2.predict(X_train).reshape(-1, 1)
    n_d, bins_d, _ = plt.hist(
        X_predicts[y_train[:, 1]==1], facecolor='green', alpha=0.5)
    n_bd, bins_bd, _ = plt.hist(
        X_predicts[y_train[:, 1]==0], facecolor='red', alpha=0.5)
    hold_out_X_train, hold_out_y_train = bd_d_dc.get_data_for_subjects(hold_out_subs)
    held_out_preds = trainer2.predict(hold_out_X_train)
    hold_out_mean_preds = []
    hold_out_classification = []
    hold_out_index = np.where(np.in1d(bd_d_dc.get_sub(), hold_out_subs))[0]
    hold_out_sub = bd_d_dc.get_sub()[hold_out_index]
    for sub_id, cat in zip(hold_out_subs, hold_out_cat):
        index = np.where(np.in1d(hold_out_sub, sub_id))[0]
        log_like = [get_log_likelihood(a, n_bd, bins_bd,
                                       n_d, bins_d) for a in held_out_preds[index]]
        prob = np.mean(log_like)
        hold_out_mean_preds.append(prob)
        hold_out_classification.append(prob>0)
    hold_out_auc = roc_auc_score(np.array(hold_out_cat), hold_out_mean_preds)
    print("hold_out AUC score = {}".format(hold_out_auc))
    cm = confusion_matrix(hold_out_cat, hold_out_classification)
    print("hold out cm = {}".format(cm))

    held_lloc_auc = roc_auc_score(hold_out_cat + mean_cats.tolist(),
                                  hold_out_mean_preds + mean_preds)
    print("held_llo AUC score = {}".format(held_lloc_auc))

    others_sub = other_dc.get_unique_subject_list()
    others_cat = other_dc.get_cat_letter_of_subject(others_sub)

    others_X_train, others_y_train = other_dc.get_data_for_subjects(others_sub)
    others_preds = trainer2.predict(others_X_train)
    test_cat_list = {}
    for sub_id, cat in zip(others_sub, others_cat):
        index = np.where(np.in1d(other_dc.get_sub(), sub_id))
        log_like = [get_log_likelihood(a, n_bd, bins_bd,
                                       n_d, bins_d) for a in others_preds[index]]
        prob = np.mean(log_like)
        if cat not in test_cat_list:
            test_cat_list[cat] = []
        test_cat_list[cat].append(prob)
    hold_out_test_cats = np.array(hold_out_cat)
    hold_out_bd_preds = np.array(hold_out_mean_preds)[np.where(hold_out_test_cats==0)[0]]
    hold_out_d_preds = np.array(hold_out_mean_preds)[np.where(hold_out_test_cats==1)[0]]
    print("mean BD {}+{}".format(np.mean(hold_out_bd_preds), np.std(hold_out_bd_preds)))
    print("mean D {}+{}".format(np.mean(hold_out_d_preds), np.std(hold_out_d_preds)))
    print("mean BR {}+{}".format(np.mean(test_cat_list["BR"]), np.std(test_cat_list["BR"])))
    print("mean R {}+{}".format(np.mean(test_cat_list["R"]), np.std(test_cat_list["R"])))
    print("mean C {}+{}".format(np.mean(test_cat_list["C"]), np.std(test_cat_list["C"])))

    remitted_auc = roc_auc_score([0]*len( test_cat_list["BR"]) + [1]*len(test_cat_list["R"]),
                                 test_cat_list["BR"] + test_cat_list["R"])
    print("Remitted AUC = {}".format(remitted_auc))

    cats = hold_out_test_cats.tolist() + mean_cats.tolist() + [0]*len(test_cat_list["BR"]) + [1]*len(test_cat_list["R"]) + [1]*len(test_cat_list["C"])
    preds = hold_out_mean_preds + mean_preds + test_cat_list["BR"] + test_cat_list["R"] + test_cat_list["C"]
    all_auc = roc_auc_score(cats, preds)
    cm = confusion_matrix(cats, np.array(preds)>0)
    print("all AUC ={}".format(all_auc))
    print("all cm = {}".format(cm))

    def ste(arr):
        return np.std(arr)/np.sqrt(len(arr))

    if plot:
        plt.clf()
        cats = hold_out_test_cats.tolist() + mean_cats.tolist()
        preds = hold_out_mean_preds + mean_preds
        plotROC(cats, preds, filename="mdd_bd_llo")
        
        plt.subplot(2, 2, 1)
        cats = hold_out_test_cats.tolist() + mean_cats.tolist() + [0]*len(test_cat_list["BR"]) + [1]*len(test_cat_list["R"])
        preds = hold_out_mean_preds + mean_preds + test_cat_list["BR"] + test_cat_list["R"]
        plotROC(cats, preds, xlabel="BD & BD-R vs MDD & MDD-R ")

        plt.subplot(2, 2, 2)
        cats = hold_out_test_cats.tolist() + mean_cats.tolist() + [0]*len(test_cat_list["BR"]) + [1]*len(test_cat_list["R"]) + [1]*len(test_cat_list["C"])
        preds = hold_out_mean_preds + mean_preds + test_cat_list["BR"] + test_cat_list["R"] + test_cat_list["C"]
        plotROC(cats, preds, xlabel="BD & BD-r vs MDD & MDD-R & C \n")

        plt.subplot(2, 2, 3)
        bd_hold_out_idx = hold_out_test_cats == 0
        bd_hold_out_cats = hold_out_test_cats[bd_hold_out_idx]
        bd_hold_mean_preds = np.array(hold_out_mean_preds)[bd_hold_out_idx] 

        bd_mean_cat_idx = mean_cats == 0
        bd_mean_cats = mean_cats[bd_mean_cat_idx]
        bd_mean_preds = np.array(mean_preds)[bd_mean_cat_idx]

        cats = bd_hold_out_cats.tolist() + bd_mean_cats.tolist() + [1]*len(test_cat_list["R"]) + [0]*len(test_cat_list["C"])
        preds = bd_hold_mean_preds.tolist() + bd_mean_preds.tolist() + test_cat_list["R"] + test_cat_list["C"]
        plotROC(cats, preds, xlabel="BD & BD-r vs C ")
        
        plt.savefig("img/rocs.png")
        plt.clf()
        from scipy.stats import ttest_ind
        x = [1, 2, 3, 4, 5]
        y = [hold_out_bd_preds, hold_out_d_preds, test_cat_list["BR"], test_cat_list["R"], test_cat_list["C"]]
        plt.boxplot(y)
        plt.xticks([1, 2, 3, 4, 5], ("BD", "MDD", "BD-R", "MDD-R", "Con"))
        plt.ylabel("Similarity index")
        plt.grid(axis='y', color="0.9", linestyle='-', linewidth=1)
        plt.xlim([0.5, 5.5])
        #plt.ylim([0.4, 1])
        plt.savefig("img/mle_bar.png")
        
        t, p = ttest_ind(hold_out_bd_preds, hold_out_d_preds)
        print("BD vs MDD t={}, p={}".format(t, p))
        t, p = ttest_ind(hold_out_d_preds, test_cat_list["R"])
        print("MDD vs R t={}, p={}".format(t, p))
        t, p = ttest_ind(hold_out_d_preds, test_cat_list["C"])
        print("MDD vs C t={}, p={}".format(t, p))
        t, p = ttest_ind(hold_out_d_preds, test_cat_list["BR"])
        print("MDD vs BR t={}, p={}".format(t, p))

        t, p = ttest_ind(hold_out_bd_preds, test_cat_list["R"])
        print("BD vs R t={}, p={}".format(t, p))
        t, p = ttest_ind(hold_out_bd_preds, test_cat_list["C"])
        print("BD vs C t={}, p={}".format(t, p))
        t, p = ttest_ind(hold_out_bd_preds, test_cat_list["BR"])
        print("BD vs BR t={}, p={}".format(t, p))

        t, p = ttest_ind(test_cat_list["C"], test_cat_list["BR"])
        print("C vs BR t={}, p={}".format(t, p))

        t, p = ttest_ind(test_cat_list["C"], test_cat_list["R"])
        print("C vs R t={}, p={}".format(t, p))


        multi_preds = np.concatenate((hold_out_bd_preds, hold_out_d_preds, test_cat_list["BR"], test_cat_list["R"], test_cat_list["C"]))
        multi_cat = [0]*len(hold_out_bd_preds) + [1]*len(hold_out_d_preds) + [2]*len(test_cat_list["BR"]) + [3]*len(test_cat_list["R"]) + [4]*len(test_cat_list["C"])
        multi_out = pd.DataFrame([multi_preds, multi_cat]).T
        multi_out.to_csv("multi_out.csv")
        import pdb; pdb.set_trace();

