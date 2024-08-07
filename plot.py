import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import itertools


class Plot:
    @staticmethod
    def show_train_history(train_history, train_metrics, validation_metrics):
        plt.plot(train_history[train_metrics])
        plt.plot(train_history[validation_metrics])
        plt.title('Train History')
        plt.ylabel(train_metrics)
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='center right')

    @staticmethod
    # 显示训练过程
    def plot_train_history(history, file_path):
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        try:
            Plot.show_train_history(history, 'accuracy', 'val_accuracy')
        except:
            Plot.show_train_history(history, 'acc', 'val_acc')
        plt.subplot(1, 2, 2)
        Plot.show_train_history(history, 'loss', 'val_loss')
        plt.savefig(file_path)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        # 自定义混淆矩阵颜色
        # cmap： color map，https://matplotlib.org/examples/color/colormaps_reference.html
        # https://scipy.github.io/old-wiki/pages/Cookbook/Matplotlib/Show_colormaps
        # custom_cm = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#7598AC'], 256) # 灰蓝
        # custom_cm = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#705286'], 100) # 紫色
        # custom_cm = matplotlib.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#304193'], 100) # 深蓝

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        # new_c = ['T'+it for it in classes]
        # plt.xticks(tick_marks, new_c)
        # plt.yticks(tick_marks, new_c)

        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                text = '%s\n%s%%' % (cm[i, j], round(cm_percent[i, j] * 100, 2))
            else:
                text = '%s' % (cm[i, j])
            plt.text(j, i, text,
                     horizontalalignment="center",
                     verticalalignment='center',
                     color="white" if cm[i, j] > thresh else "black")

        # 去掉该语句，否则保存的混淆矩阵图左侧文字显示不全
        # plt.tight_layout()
        plt.ylabel('Real-IHC')
        plt.xlabel('AI-IHC')

    @staticmethod
    def show_matrix(y_pred, y_true, classes_count, out_put_file, normalize=True, fig_size=4, dpi=110, savefig=True,
                    title='Confusion matrix', classes=None):
        cnf_matrix = confusion_matrix(y_true, y_pred)
        # Plot non-normalized confusion matrix

        plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
        if classes is None:
            classes = [str(x) for x in range(classes_count)]
        Plot.plot_confusion_matrix(cnf_matrix, classes=classes, title=title, normalize=normalize)

        err = 0
        for i in range(0, len(y_pred)):
            if y_pred[i] != y_true[i]:
                err += 1

        overall_acc = 1 - err * 1.0 / len(y_pred)
        print(cnf_matrix)
        acc_list = []
        for i in range(cnf_matrix.shape[0]):
            acc = 100 * cnf_matrix[i, i] / np.sum(cnf_matrix[i, :])
            print('%02d acc: %.2f%%' % (i, acc))
            acc_list.append(acc)

        overall = 100 * overall_acc
        mean = np.mean(acc_list)
        print('overall acc: %.2f%%, avg acc: %.2f%%' % (overall, mean))
        print("accurracy:{}".format(overall_acc))

        if savefig:
            save_fn = out_put_file + 'confusion_matrix_%.2f_%.2f.png' % (overall, mean)
            plt.savefig(save_fn, dpi=dpi)
            print("save plot image to %s " % save_fn)

    @staticmethod
    def plot_roc(y_true, predicts, fn=None):
        fpr, tpr, thresholds = roc_curve(y_true, predicts)
        roc_auc = round(auc(fpr, tpr), 4)
        if fn is None:
            return roc_auc
        fs = 16

        plt.subplots(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=fs)
        plt.yticks(fontsize=fs)
        plt.xlabel('False Positive Rate', fontsize=fs)
        plt.ylabel('True Positive Rate', fontsize=fs)
        plt.title('ROC Curve', fontsize=fs)
        plt.legend(loc="lower right", fontsize=fs)
        plt.savefig(fn + '%s.jpg' % roc_auc)
        plt.show()
        return roc_auc
