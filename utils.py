import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from visdom import Visdom
import torch
from torchvision.utils import save_image
import math
import os
import json

mean = 112.75457
sd = 97.87436
mn = -40.0
mx = 347.0


def poly_lr(epoch, max_epochs, initial_lr, exponent=0.9):
    return initial_lr * (1 - epoch / max_epochs) ** exponent


def save_roc_curve(true_labels, output_scores, figure_name='roc.png'):
    avg_auc = 0

    for i in range(output_scores.shape[1]):
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i)
        auc = metrics.auc(fpr, tpr)
        avg_auc += auc
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.plot(fpr, tpr, label='C: ' + str(i) + ', AUC: ' + '{:.3f}'.format(auc))

    avg_auc /= output_scores.shape[1]
    plt.title('avg. AUC: ' + '{:.3f}'.format(avg_auc))
    plt.legend()
    plt.savefig(figure_name)
    plt.cla()


def save_roc_curve_bin(true_labels, output_scores, figure_name='roc.png'):
    fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
    auc = metrics.auc(fpr, tpr)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc))

    plt.title('avg. AUC: ' + '{:.3f}'.format(auc))
    plt.legend()
    plt.savefig(figure_name)
    plt.cla()


def find_opt_thr(true_labels, output_scores):
    opt_thr = np.array([])
    for i in range(output_scores.shape[1]):
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i, drop_intermediate=False)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = np.append(opt_thr, thr[best_idx])

    return opt_thr


def get_pr_auc(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']

    if num_classes is 2:
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc * 100))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(num_classes):
            # pre, rec, thr = metrics.precision_recall_curve(true_labels, output_scores[:, i], pos_label=i)
            ap = metrics.average_precision_score((true_labels == i).astype(np.int), output_scores[:, i])

            auc = np.append(auc, ap)
        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))

    return auc


def get_roc_auc(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']

    if num_classes is 2:
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc * 100))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(num_classes):
            fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i, drop_intermediate=False)
            auc = np.append(auc, metrics.auc(fpr, tpr))
            best_idx = np.argmax(1 - fpr + tpr)
            opt_thr = np.append(opt_thr, thr[best_idx])
            if save_roc and not math.isnan(auc[i]):
                # plt.plot(fpr * 100, tpr * 100, label='C: ' + str(i) + ', AUC: ' + '{:.1f}'.format(auc[i] * 100))
                plt.plot(fpr, tpr, label=class_name[i] + ': AUC: ' + '{:.1f}'.format(auc[i]))
        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))
        if save_roc:
            plt.title('Internal DB (avg. AUC: ' + '{:.1f})'.format(auc[num_classes]))
    if save_roc:
        plt.xlabel('1 - specificity (%)')
        plt.ylabel('Sensitivity (%)')
        plt.legend()
        plt.savefig(figure_name)
        plt.cla()

    if find_opt_thr:
        return auc, opt_thr

    return auc


def eval_multi_class(true_labels, output_scores, dec_thr=None):
    num_classes = output_scores.shape[1]

    sen = np.array([])
    spe = np.array([])
    acc = np.array([])

    for cls_idx in range(output_scores.shape[1]):
        pos_true = true_labels == cls_idx
        neg_true = np.invert(pos_true)

        pos_pred = output_scores[:, cls_idx] >= dec_thr[cls_idx]
        neg_pred = np.invert(pos_pred)

        tp = np.logical_and(pos_pred, pos_true)
        tn = np.logical_and(neg_pred, neg_true)

        tpr = np.sum(tp) / np.sum(pos_true)
        tnr = np.sum(tn) / np.sum(neg_true)

        sen = np.append(sen, tpr)
        spe = np.append(spe, tnr)

    sen = np.append(sen, np.nanmean(sen))
    spe = np.append(spe, np.nanmean(spe))

    return sen, spe


def get_sen_spe_from_conf_mat(conf_mat):
    num_classes = conf_mat.shape[0]

    sen = np.array([])
    spe = np.array([])
    acc = np.array([])

    for cls_idx in range(num_classes):
        tp = conf_mat[cls_idx, cls_idx]
        pos_trues = np.sum(conf_mat[cls_idx, :])
        pos = np.sum(conf_mat[:, cls_idx])

        neg_trues = np.sum(conf_mat) - pos_trues
        tn = neg_trues - pos + tp

        tpr = tp / pos_trues
        tnr = tn / neg_trues

        sen = np.append(sen, tpr)
        spe = np.append(spe, tnr)

    sen = np.append(sen, np.nanmean(sen))
    spe = np.append(spe, np.nanmean(spe))

    return sen, spe

def eval_pr_multi_class(true_labels, output_scores, dec_thr=None):
    num_classes = output_scores.shape[1]

    sen = np.array([])
    spe = np.array([])
    acc = np.array([])

    for cls_idx in range(output_scores.shape[1]):
        pos_true = true_labels == cls_idx
        neg_true = np.invert(pos_true)

        pos_pred = output_scores[:, cls_idx] >= dec_thr[cls_idx]
        neg_pred = np.invert(pos_pred)

        tp = np.logical_and(pos_pred, pos_true)
        tn = np.logical_and(neg_pred, neg_true)

        tpr = np.sum(tp) / np.sum(pos_pred)
        tnr = np.sum(tp) / np.sum(pos_true)

        print(np.sum(pos_true) == (np.sum(tp) + np.sum(neg_true) - np.sum(tn)))

        sen = np.append(sen, tpr)
        spe = np.append(spe, tnr)

    sen = np.append(sen, np.nanmean(sen))
    spe = np.append(spe, np.nanmean(spe))

    return sen, spe

def eval_multi_class_index(true_labels, pred, dec_thr=None):
    sen = np.array([])
    spe = np.array([])
    acc = np.array([])
    # num_classes = len(np.unique(true_labels))
    num_classes = 5
    for cls_idx in range(num_classes):
        pos_true = true_labels == cls_idx
        neg_true = np.invert(pos_true)

        pos_pred = pred == cls_idx
        neg_pred = np.invert(pos_pred)

        tp = np.sum(np.logical_and(pos_pred, pos_true))
        tn = np.sum(np.logical_and(neg_pred, neg_true))

        fn = np.sum(pos_true) - np.sum(tp)
        fp = np.sum(neg_true) - np.sum(tn)

        # tpr = np.sum(tp) / np.sum(pos_true)
        # tnr = np.sum(tn) / np.sum(neg_true)

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        # tpr = tp / (tp + fp)
        # tnr = tp / (tp + fn)

        sen = np.append(sen, tpr)
        spe = np.append(spe, tnr)

    sen = np.append(sen, np.nanmean(sen))
    spe = np.append(spe, np.nanmean(spe))

    return sen, spe


def eval_multi_class_index_two(true_labels, pred, dec_thr=None):
    sen = np.array([])
    spe = np.array([])
    acc = np.array([])
    # num_classes = len(np.unique(true_labels))
    num_classes = 5

    for cls_idx in range(num_classes):

        pos_true = true_labels == cls_idx
        neg_true = np.invert(pos_true)

        '''
        pos_pred = np.array([], dtype=np.bool)
        for i in range(len(true_labels)):
            if pred[i, 0] == cls_idx:
                if true_labels[i] == cls_idx:
                    pos_pred = np.append(pos_pred, True)
                elif true_labels[i] == pred[i, 1]:
                    pos_pred = np.append(pos_pred, False)
                else:
                    pos_pred = np.append(pos_pred, True)
            else:
                if true_labels[i] != cls_idx:
                    pos_pred = np.append(pos_pred, False)
                elif true_labels[i] == pred[i, 1]:
                    pos_pred = np.append(pos_pred, True)
                else:
                    pos_pred = np.append(pos_pred, False)
        '''
        pos_pred = np.sum(pred == cls_idx, axis=1) > 0
        neg_pred = np.invert(pos_pred)

        tp = np.sum(np.logical_and(pos_pred, pos_true))
        tn = np.sum(np.logical_and(neg_pred, neg_true))
        fn = np.sum(pos_true) - np.sum(tp)
        fp = np.sum(neg_true) - np.sum(tn)

        # tpr = np.sum(tp) / np.sum(pos_true)
        # tnr = np.sum(tn) / np.sum(neg_true)

        tpr = tp / (tp + fn)
        tnr = tn / (tn + fp)

        # tpr = tp / (tp + fp)
        # tnr = tp / (tp + fn)

        sen = np.append(sen, tpr)
        spe = np.append(spe, tnr)

    sen = np.append(sen, np.nanmean(sen))
    spe = np.append(spe, np.nanmean(spe))

    return sen, spe


def get_radiol_results():
    root_dir = '/data/ESMH/radiologist'
    files = os.listdir(root_dir)
    files = [file for file in files if file.endswith(".json")]
    files = sorted(files)

    all_data = []
    for file in files:
        file_path = os.path.join(root_dir, file)
        with open(file_path) as json_file:
            all_data.append(json.load(json_file))

    with open('/data/ESMH/test_3_phase.json') as json_file:
        cases = json.load(json_file)

    with open('/data/ESMH/subtypes.json') as json_file:
        labels = json.load(json_file)

    true_labels = np.array([])
    for _, (idx, case_id) in enumerate(cases.items()):
        true_labels = np.append(true_labels, labels[case_id])

    all_predictions = []
    for data in all_data:
        predictions = np.array([]).reshape(0, 2)
        for _, (idx, values) in enumerate(data.items()):
            if values[1] == None:
                values[1] = values[0]
            values = np.expand_dims(np.array(values), axis=0)
            predictions = np.append(predictions, values, axis=0)

        all_predictions.append(predictions)

    sen_best = np.array([]).reshape(0, 6)
    spe_best = np.array([]).reshape(0, 6)
    for prediction in all_predictions:
        pred_best = prediction[:, 0]
        sen, spe = eval_multi_class_index(true_labels, pred_best)
        sen = np.expand_dims(sen, axis=0)
        spe = np.expand_dims(spe, axis=0)
        sen_best = np.append(sen_best, sen, axis=0)
        spe_best = np.append(spe_best, spe, axis=0)

    avg_sen_best = np.nanmean(sen_best, axis=0, keepdims=True)
    avg_spe_best = np.nanmean(spe_best, axis=0, keepdims=True)
    sen_best = np.append(sen_best, avg_sen_best, axis=0)
    spe_best = np.append(spe_best, avg_spe_best, axis=0)
    '''
    # best voting
    pred_best_stack = np.array([]).reshape(len(true_labels), 0)
    for prediction in all_predictions:
        pred_best_stack = np.append(pred_best_stack, prediction[:, 0:1], axis=1)

    pred_best_voting = np.array([])
    for data in pred_best_stack:
        data = data.astype(np.int)
        counts = np.bincount(data)
        pred = np.argmax(counts)
        pred_best_voting = np.append(pred_best_voting, pred)
    
    sen_best_voting, spe_best_voting = eval_multi_class_index(true_labels, pred_best_voting)
    sen_best_voting = np.expand_dims(sen_best_voting, axis=0)
    spe_best_voting = np.expand_dims(spe_best_voting, axis=0)
    sen_best = np.append(sen_best, sen_best_voting, axis=0)
    spe_best = np.append(spe_best, spe_best_voting, axis=0)
    '''
    # print('sen_best', sen_best)
    # print('spe_best', spe_best)

    sen_two = np.array([]).reshape(0, 6)
    spe_two = np.array([]).reshape(0, 6)
    for prediction in all_predictions:
        sen, spe = eval_multi_class_index_two(true_labels, prediction)
        sen = np.expand_dims(sen, axis=0)
        spe = np.expand_dims(spe, axis=0)
        sen_two = np.append(sen_two, sen, axis=0)
        spe_two = np.append(spe_two, spe, axis=0)

    avg_sen_two = np.nanmean(sen_two, axis=0, keepdims=True)
    avg_spe_two = np.nanmean(spe_two, axis=0, keepdims=True)
    sen_two = np.append(sen_two, avg_sen_two, axis=0)
    spe_two = np.append(spe_two, avg_spe_two, axis=0)
    # print('sen_two', sen_two)
    # print('spe_two', spe_two)

    return sen_best, spe_best, sen_two, spe_two


def get_radiol_results_ordered():
    root_dir = '/data/ESMH/radiologist'
    files = os.listdir(root_dir)
    files = [file for file in files if file.endswith(".json")]
    files = sorted(files)

    all_data = []
    for file in files:
        file_path = os.path.join(root_dir, file)
        with open(file_path) as json_file:
            all_data.append(json.load(json_file))

    with open('/data/ESMH/test_3_phase.json') as json_file:
        cases = json.load(json_file)

    with open('/data/ESMH/subtypes.json') as json_file:
        labels = json.load(json_file)

    true_labels = np.array([])
    for _, (idx, case_id) in enumerate(cases.items()):
        true_labels = np.append(true_labels, labels[case_id])

    case_ids = np.array(list(cases.values()))
    sort_idx = np.argsort(case_ids)
    true_labels = true_labels[sort_idx]

    all_predictions = []
    for data in all_data:
        predictions = np.array([]).reshape(0, 2)
        for _, (idx, values) in enumerate(data.items()):
            if values[1] == None:
                values[1] = values[0]
            values = np.expand_dims(np.array(values), axis=0)
            predictions = np.append(predictions, values, axis=0)

        predictions = predictions[sort_idx]
        all_predictions.append(predictions)

    return all_predictions


def get_pr_auc_fig(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']
    font_size = 9

    # [0.13, 0.31, 0.39, 0.42, 0.15]
    model_pre, model_rec = eval_pr_multi_class(true_labels, output_scores, dec_thr=[0.14, 0.34, 0.43, 0.31, 0.37])

    # ci_l = [0.8106, 0.8381, 0.6584, 0.8205, 0.6973]
    # ci_h = [0.9822, 0.9963, 0.9626, 0.9759, 0.9892]

    # tcia 40 test
    # ci_l = [0., 0., 0.554, 0.762, 0.805]
    # ci_h = [0., 0., 0.961, 0.970, 0.997]


    # full test
    # ci_l = [0., 0., 0.521, 0.807, 0.835]
    # ci_h = [0., 0., 0.952, 0.955, 0.960]

    # esmh ap
    # ci_l = [0.3158, 0.4567, 0.4216, 0.465, 0.590]
    # ci_h = [0.9251, 0.9683, 0.9113, 0.924, 0.963]

    # full tcia AP
    ci_l = [0., 0., 0.046, 0.236, 0.972]
    ci_h = [0., 0., 0.621, 0.738, 0.996]

    sen_rad_best, spe_rad_best, sen_rad_two, spe_rad_two = get_radiol_results()

    if num_classes is 2:
        fpr, tpr, thr = metrics.precision_recall_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc * 100))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(2, num_classes):
            pre, rec, thr = metrics.precision_recall_curve(true_labels, output_scores[:, i], pos_label=i)
            ap = metrics.average_precision_score((true_labels == i).astype(np.int), output_scores[:, i])
            if save_roc and not math.isnan(ap):
                plt.grid(linestyle='--')
                plt.axes().set_axisbelow(True)
                plt.axes().set_aspect('equal')
                plt.title(class_name[i], fontsize=font_size)
                plt.plot(rec, pre, label='Model: AP=' + '{:.3f}'.format(ap) + '\n' + '(' + '{:.3f}'.format(ci_l[i]) + '{0}'.format(u'\u2013') + '{:.3f})'.format(ci_h[i]), zorder=1)
                # plt.plot(rec, pre, label='Model: AP=' + '{:.3f}'.format(ap), zorder=1)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.xlabel('Recall', fontsize=font_size)
                plt.ylabel('Precision', fontsize=font_size)

                loc = 3

                if i == 1:
                    loc = 3
                if i == 3:
                    loc = 3
                if i == 4:
                    loc = 3

                plt.legend(loc=loc, fontsize=font_size)

                # plt.scatter(model_rec[i], model_pre[i], label='Operating point', marker='d', color='c', zorder=3)
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.ylim((-0.05, 1.05))
                plt.savefig(os.path.join(figure_name, 'pr_' + class_name[i] + '.svg'), bbox_inches='tight', format='svg')
                plt.cla()
                continue

                plt.scatter(spe_rad_best[:6, i],
                            sen_rad_best[:6, i], label='Individual radiologists (top-1)', marker='x', color='m', zorder=2)

                # for idx in range(6):
                #     plt.text(1 - spe_rad_best[idx, i], sen_rad_best[idx, i], str(idx+1))

                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.scatter(spe_rad_two[:6, i],
                            sen_rad_two[:6, i], label='Individual radiologists (top-2)', marker='*', color='y', zorder=2)
                # for idx in range(6):
                #     plt.text(1 - spe_rad_two[idx, i], sen_rad_two[idx, i], str(idx+1))
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)


                plt.scatter(spe_rad_best[6, i], sen_rad_best[6, i], label='Average radiologist (top-1)', marker='d',
                            color='m', zorder=2)

                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.scatter(spe_rad_two[6, i], sen_rad_two[6, i], label='Average radiologist (top-2)', marker='d',
                            color='y', zorder=2)
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.ylim((-0.05, 1.05))
                plt.savefig(os.path.join(figure_name, 'pr_' + class_name[i] + '.svg'), bbox_inches='tight', format='svg')
                # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                plt.cla()

        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))

    if find_opt_thr:
        return auc, opt_thr

    return auc


def get_roc_auc_fig(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']
    font_size = 19

    # ci_l = [0.8106, 0.8381, 0.6584, 0.8205, 0.6973]
    # ci_h = [0.9822, 0.9963, 0.9626, 0.9759, 0.9892]

    # tcia 40 test
    # ci_l = [0., 0., 0.554, 0.762, 0.805]
    # ci_h = [0., 0., 0.961, 0.970, 0.997]

    # full test
    ci_l = [0., 0., 0.521, 0.807, 0.835]
    ci_h = [0., 0., 0.952, 0.955, 0.960]

    sen_rad_best, spe_rad_best, sen_rad_two, spe_rad_two = get_radiol_results()

    if num_classes is 2:
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc * 100))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(num_classes):
            fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i)
            auc = np.append(auc, metrics.auc(fpr, tpr))
            best_idx = np.argmax(1 - fpr + tpr)
            opt_thr = np.append(opt_thr, thr[best_idx])
            if save_roc and not math.isnan(auc[i]):
                plt.grid(linestyle='--')
                plt.axes().set_axisbelow(True)
                plt.axes().set_aspect('equal')
                plt.title(class_name[i], fontsize=font_size)
                # plt.plot(fpr, tpr, label='Model: AUC=' + '{:.3f}'.format(auc[i]) + '\n' + '(95% CI: ' + '{:.3f}'.format(ci_l[i]) + '{0}'.format(u'\u2013') + '{:.3f})'.format(ci_h[i]), zorder=1)
                plt.plot(fpr, tpr, label='AUC=' + '{:.3f}'.format(auc[i]) + '\n' + '(' + '{:.3f}'.format(
                    ci_l[i]) + '{0}'.format(u'\u2013') + '{:.3f})'.format(ci_h[i]), zorder=1)

                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.xlabel('1 - Specificity', fontsize=font_size)
                plt.ylabel('Sensitivity', fontsize=font_size)

                if i == 2:
                    loc = 4
                else:
                    loc = 4
                plt.legend(loc=loc, fontsize=font_size)

                # plt.scatter(1 - spe[i], sen[i], label='Model operating point', marker='d', color='c', zorder=3)
                # plt.scatter(1 - spe[4-i], sen[4-i], label='Operating point', marker='d', color='c', zorder=3)
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                # plt.savefig(os.path.join(figure_name, class_name[i] + '.svg'), bbox_inches='tight', format='svg')
                # plt.cla()
                # continue

                plt.scatter(1 - spe_rad_best[:6, i],
                            sen_rad_best[:6, i], label='Individual radiologists (top-1)', marker='x', color='m', zorder=2)

                # for idx in range(6):
                #     plt.text(1 - spe_rad_best[idx, i], sen_rad_best[idx, i], str(idx+1))

                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.scatter(1 - spe_rad_two[:6, i],
                            sen_rad_two[:6, i], label='Individual radiologists (top-2)', marker='*', color='y', zorder=2)
                # for idx in range(6):
                #     plt.text(1 - spe_rad_two[idx, i], sen_rad_two[idx, i], str(idx+1))
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)


                plt.scatter(1 - spe_rad_best[6, i], sen_rad_best[6, i], label='Average radiologist (top-1)', marker='d',
                            color='m', zorder=2)

                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.scatter(1 - spe_rad_two[6, i], sen_rad_two[6, i], label='Average radiologist (top-2)', marker='d',
                            color='y', zorder=2)
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.savefig(os.path.join(figure_name, class_name[i] + '.svg'), bbox_inches='tight', format='svg')
                # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                plt.cla()

        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))

    if find_opt_thr:
        return auc, opt_thr

    return auc


def get_roc_auc_fig_two(true_labels, true_labels_, output_scores, output_scores_, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']
    font_size = 9

    # [0.13, 0.31, 0.39, 0.42, 0.15]
    # sen, spe = eval_multi_class(true_labels, output_scores, dec_thr=[0.14, 0.34, 0.43, 0.31, 0.37])
    conf_mat = calc_conf_mat(true_labels, output_scores, thr=[0.10, 0.35, 0.45, 0.30, 0.39])
    sen, spe = get_sen_spe_from_conf_mat(conf_mat)

    ci_l_ = [0.8106, 0.8381, 0.6584, 0.8205, 0.6973]
    ci_h_ = [0.9822, 0.9963, 0.9626, 0.9759, 0.9892]

    # tcia 40 test
    # ci_l_ = [0., 0., 0.554, 0.762, 0.805]
    # ci_h_ = [0., 0., 0.961, 0.970, 0.997]

    # full test
    ci_l = [0., 0., 0.521, 0.807, 0.835]
    ci_h = [0., 0., 0.952, 0.955, 0.960]

    sen_rad_best, spe_rad_best, sen_rad_two, spe_rad_two = get_radiol_results()

    if num_classes is 2:
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc * 100))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(2, num_classes):
            fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i)
            fpr_, tpr_, thr_ = metrics.roc_curve(true_labels_, output_scores_[:, i], pos_label=i)
            auc =  metrics.auc(fpr, tpr)
            auc_ = metrics.auc(fpr_, tpr_)
            best_idx = np.argmax(1 - fpr + tpr)
            opt_thr = np.append(opt_thr, thr[best_idx])
            if save_roc and not math.isnan(auc):
                plt.grid(linestyle='--')
                plt.axes().set_axisbelow(True)
                plt.axes().set_aspect('equal')
                plt.title(class_name[i], fontsize=font_size)
                plt.plot(fpr_, tpr_, label='Model: AUC=' + '{:.3f}'.format(auc_) + '\n' + '(95% CI: ' + '{:.3f}'.format(ci_l_[i]) + '{0}'.format(u'\u2013') + '{:.3f})'.format(ci_h_[i]), zorder=1)
                # plt.plot(fpr_, tpr_, label='Model AUC=' + '{:.3f}'.format(auc_) + '\n' + '(' + '{:.3f}'.format(
                #     ci_l_[i]) + '{0}'.format(u'\u2013') + '{:.3f})'.format(ci_h_[i]), zorder=1)

                if i == 2:
                    loc = 1
                else:
                    loc = 4
                plt.legend(loc=loc, fontsize=font_size)

                plt.plot(fpr, tpr, label='Model (TCIA): AUC=' + '{:.3f}'.format(auc) + '\n' + '(95% CI: ' + '{:.3f}'.format(
                    ci_l[i]) + '{0}'.format(u'\u2013') + '{:.3f})'.format(ci_h[i]), zorder=0)


                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.xlabel('1 - Specificity', fontsize=font_size)
                plt.ylabel('Sensitivity', fontsize=font_size)

                plt.legend(loc=loc, fontsize=font_size)

                # plt.scatter(1 - spe[i], sen[i], label='Model operating point', marker='d', color='c', zorder=3)
                # plt.scatter(1 - spe[4-i], sen[4-i], label='Operating point', marker='d', color='c', zorder=3)
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                # plt.savefig(os.path.join(figure_name, class_name[i] + '.svg'), bbox_inches='tight', format='svg')
                # plt.cla()
                # continue

                plt.scatter(1 - spe_rad_best[:6, i],
                            sen_rad_best[:6, i], label='Individual radiologists (top-1)', marker='x', color='m', zorder=2)

                # for idx in range(6):
                #     plt.text(1 - spe_rad_best[idx, i], sen_rad_best[idx, i], str(idx+1))

                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.scatter(1 - spe_rad_two[:6, i],
                            sen_rad_two[:6, i], label='Individual radiologists (top-2)', marker='*', color='y', zorder=2)
                # for idx in range(6):
                #     plt.text(1 - spe_rad_two[idx, i], sen_rad_two[idx, i], str(idx+1))
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)


                plt.scatter(1 - spe_rad_best[6, i], sen_rad_best[6, i], label='Average radiologist (top-1)', marker='d',
                            color='m', zorder=2)

                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.scatter(1 - spe_rad_two[6, i], sen_rad_two[6, i], label='Average radiologist (top-2)', marker='d',
                            color='y', zorder=2)
                if i > -1:
                    plt.legend(loc=loc, fontsize=font_size)

                plt.savefig(os.path.join(figure_name, class_name[i] + '.svg'), bbox_inches='tight', format='svg')
                # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                plt.cla()

        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))

    if find_opt_thr:
        return auc, opt_thr

    return auc


def get_roc_auc_fig_backup(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']
    font_size = 7
    if num_classes is 2:
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc * 100))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(num_classes):
            fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i)
            auc = np.append(auc, metrics.auc(fpr, tpr))
            best_idx = np.argmax(1 - fpr + tpr)
            opt_thr = np.append(opt_thr, thr[best_idx])
            if save_roc and not math.isnan(auc[i]):
                plt.figure(figsize=(3.3, 3.5))
                plt.grid(linewidth=1, linestyle='--')
                plt.axes().set_axisbelow(True)
                plt.setp(plt.axes().get_xticklabels(), visible=False)
                plt.setp(plt.axes().get_yticklabels(), visible=False)
                plt.axes().set_aspect('equal')
                # plt.title(class_name[i], fontsize=font_size)
                # plt.plot(fpr, tpr, label='Model: AUC= ' + '{:.3f}'.format(auc[i]), linewidth=2)
                # plt.plot(fpr, tpr, label='Model: AUC= ' + '{:.3f}'.format(auc[i]), linewidth=2)
                plt.plot(fpr, tpr, label='m')
                # plt.xticks(fontsize=font_size)
                # plt.yticks(fontsize=font_size)
                # plt.xlabel('1 - Specificity', fontsize=font_size, fontname='Arial')
                # plt.ylabel('Sensitivity', fontsize=font_size)
                plt.legend(loc=4, fontsize=font_size)

                plt.scatter([0.3, 0.3, 0.35, 0.35, 0.4, 0.4],
                            [0.6, 0.65, 0.6, 0.7, 0.65, 0.7], label='I', marker='+', color='m')
                plt.legend(loc=4, fontsize=font_size)

                plt.scatter(0.35, 0.65, label='A', marker='o', color='g')
                plt.legend(loc=4, fontsize=font_size)

                # plt.savefig(os.path.join(figure_name, class_name[i] + '.png'), bbox_inches='tight', dpi=1200)
                # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                plt.savefig(os.path.join(figure_name, class_name[i] + '.svg'), bbox_inches='tight')
                plt.cla()

        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))

    if find_opt_thr:
        return auc, opt_thr

    return auc


def form_conf_mat(true_labels, pred_labels):
    num_classes = 5
    num_gt_classes = len(np.unique(true_labels))

    conf_mat = np.zeros((num_gt_classes, num_classes))
    true_labels = true_labels.astype(np.int)
    pred_labels = pred_labels.astype(np.int)

    for i in range(pred_labels.shape[0]):
        conf_mat[4-true_labels[i], 4-pred_labels[i]] += 1

    return conf_mat


def calc_acc_from_conf_mat(conf_mat):
    return np.sum(np.diag(conf_mat)) / np.sum(conf_mat)


def calc_top1_acc_from_scores(true_labels, scores):
    true_labels = true_labels.astype(np.int)
    count = 0
    for i in range(len(true_labels)):
        if true_labels[i] in np.argsort(scores[i])[-1:]:
            count += 1

    return count / len(true_labels)


def calc_top2_acc_from_scores(true_labels, scores):
    true_labels = true_labels.astype(np.int)
    count = 0
    for i in range(len(true_labels)):
        if true_labels[i] in np.argsort(scores[i])[-2:]:
            count += 1

    return count / len(true_labels)


def calc_top2_acc(true_labels, pred_labels):
    true_labels = true_labels.astype(np.int)
    pred_labels = pred_labels.astype(np.int)
    count = 0
    for i in range(len(true_labels)):
        if true_labels[i] in pred_labels[i]:
            count += 1

    return count / len(true_labels)


def calc_conf_mat(true_labels, output_scores, thr=None):
    true_labels = true_labels.astype(np.int)
    num_gt_classes = len(np.unique(true_labels))
    num_classes = output_scores.shape[1]

    conf_mat = np.zeros((num_gt_classes, num_classes))
    # conf_mat = np.zeros((num_classes, num_classes))
    predicted = np.argmax(output_scores, axis=1)

    # for i in range(predicted.shape[0]):
    #     if thr is not None:
    #         over_idx = np.where(output_scores[i] >= thr)[0]
    #         if len(over_idx) > 1:
    #             best_idx = np.argmax(output_scores[i][over_idx])
    #             conf_mat[true_labels[i], over_idx[best_idx]] += 1
    #         elif len(over_idx) < 1:
    #             conf_mat[true_labels[i], predicted[i]] += 1
    #         else:
    #             conf_mat[true_labels[i], over_idx] += 1
    #     else:
    #         conf_mat[true_labels[i], predicted[i]] += 1

    for i in range(predicted.shape[0]):
        if thr is not None:
            over_idx = np.where(output_scores[i] >= thr)[0]
            if len(over_idx) > 1:
                best_idx = np.argmax(output_scores[i][over_idx])
                conf_mat[4-true_labels[i], 4-over_idx[best_idx]] += 1
            elif len(over_idx) < 1:
                conf_mat[4-true_labels[i], 4-predicted[i]] += 1
            else:
                conf_mat[4-true_labels[i], 4-over_idx] += 1
        else:
            conf_mat[4-true_labels[i], 4-predicted[i]] += 1

    return conf_mat


def calc_eval_metric(conf_mat, macro=False):
    num_classes = conf_mat.shape[0]

    if num_classes is 2:
        spe, sen = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
        if macro is False:
            acc = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
        else:
            acc = (spe + sen) / 2
    else:
        sen = np.array([])
        spe = np.array([])
        acc = np.array([])
        for i in range(num_classes):
            tp = conf_mat[i, i]
            fp = np.sum(conf_mat[i, :]) - tp
            fn = np.sum(conf_mat[:, i]) - tp
            tn = np.sum(conf_mat) - tp - fp - fn

            sen = np.append(sen, tp / (tp + fn))
            spe = np.append(spe, tn / (tn + fp))
            acc = np.append(acc, (tp + tn) / (tp + fp + fn + tn))

        if macro is False:
            acc = np.append(acc, np.sum(np.diag(conf_mat)) / np.sum(conf_mat))
        else:
            acc = np.append(acc, np.sum(np.diag(conf_mat)) / np.sum(conf_mat))

    return sen, spe, acc


def calc_cls_conf_mat(true_labels, output_scores, thr=None):
    num_classes = output_scores.shape[1]
    if thr is None:
        thr = 0.5 * np.ones(num_classes)

    cls_conf_mat = np.zeros((output_scores.shape[1], 2, 2))

    for i in range(num_classes):
        y_true = true_labels == i
        y_false = np.invert(y_true)
        pos = output_scores[:, i] >= thr[i]
        tp = np.sum(np.logical_and(y_true, pos))
        fp = np.sum(pos) - tp

        neg = np.invert(pos)
        tn = np.sum(np.logical_and(y_false, neg))
        fn = np.sum(neg) - tn

        cls_conf_mat[i][0][0] = tn
        cls_conf_mat[i][0][1] = fn
        cls_conf_mat[i][1][0] = fp
        cls_conf_mat[i][1][1] = tp

    return cls_conf_mat


def calc_cls_eval_metric(cls_conf_mat):
    cls_spe = np.array([])
    cls_sen = np.array([])
    cls_acc = np.array([])

    for i in range(cls_conf_mat.shape[0]):
        conf_mat = cls_conf_mat[i]
        spe, sen = np.diag(conf_mat) / np.sum(conf_mat, axis=0)
        acc = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)

        cls_spe = np.append(cls_spe, spe)
        cls_sen = np.append(cls_sen, sen)
        cls_acc = np.append(cls_acc, acc)

    return cls_sen, cls_spe, cls_acc


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')


def save_numpy(data, file_name='img.png', is_seg=False, is_3d=False):
    data_tensor = torch.from_numpy(data.copy())
    if not is_seg:
        if is_3d:
            save_image((data_tensor.transpose(0, 1) * sd + mean - mn) / (-mn + mx), file_name)
        else:
            save_image((data_tensor.unsqueeze(1) * sd + mean - mn) / (-mn + mx), file_name)
    else:
        if is_3d:
            save_image(data_tensor.transpose(0, 1), file_name)
        else:
            save_image(data_tensor.unsqueeze(1), file_name)
