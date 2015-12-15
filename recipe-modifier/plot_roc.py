import cPickle as pickle
import matplotlib.pyplot as plt
import numpy as np

with open('save_data.p','rb') as f:
    saved_data = pickle.load(f)
    tp,fp,tn,fn,indices = saved_data
    fpr = {}
    tpr = {}

    accuracy = {}
    precision = {}
    recall = {}
    selectivity = {}
    for thresh in tp:
        np_tp = np.array(tp[thresh])
        np_fp = np.array(fp[thresh])
        np_tn = np.array(tn[thresh])
        np_fn = np.array(fn[thresh])


        total = np_fp+np_tn+np_tp+np_fn
        guessed_positives = np_tp
        total_positives = np_tp + np_fn
        accuracy[thresh] = np.true_divide(np_tp+np_tn, total)
        precision[thresh] = np.true_divide(guessed_positives, np_tp+np_fp)
        recall[thresh] = np.true_divide(np_tp, np_fn+np_tp)
    for i in range(3):
        print "TOP %d:"%(i+1)
        max_precision = 0
        max_accuracy = 0
        max_recall = 0
        max_selectivity = 0
        for thresh in tp:
            if accuracy[thresh][i] > max_accuracy:
                max_accuracy = accuracy[thresh][i]
            if precision[thresh][i] > max_precision:
                max_precision = precision[thresh][i]
            if recall[thresh][i] > max_recall:
                max_recall = recall[thresh][i]
        print 'precision:', max_precision
        print 'accuracy:', max_accuracy
        print 'recall:', max_recall
