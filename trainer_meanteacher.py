import os
import numpy as np
from PIL import Image
import shutil
import time
import datetime

import torch
import torchvision.transforms as transforms

from libs.mean_teacher_master.pytorch.mean_teacher.run_context import RunContext
from libs.mean_teacher_master.pytorch.mean_teacher.cli import parse_dict_args
from libs.mean_teacher_master.pytorch import main as mean_teacher_main
from libs.mean_teacher_master.pytorch.mean_teacher import data

from plot import Plot


def get_time_stamp():
    d = datetime.datetime.fromtimestamp(time.time())
    stamp_str = d.strftime("%Y%m%d_%H%M%S_%f")[0:15]
    return stamp_str


def extract_pred_error_files(file_test, y_pred, y_true, target_fold, extract_all=False, conf=None, print_err=False):
    if os.path.exists(target_fold):
        shutil.rmtree(target_fold)
    move_files = 0
    for i in range(0, len(file_test)):
        if y_true[i] != y_pred[i] or extract_all:
            file = file_test[i]
            if os.path.exists(file):
                dirs = target_fold + '/' + str(int(y_true[i])) + '/' + str(int(y_pred[i])) + '/'
                if not os.path.exists(dirs):
                    os.makedirs(dirs)

                fn = os.path.basename(file)
                dest = dirs + fn
                move_files += 1
                if conf is not None:
                    src_img = read_img_file(file_test[i])
                    # if len(self.desc_list) == 2:
                    if len(conf[i]) == 1:
                        conf_str = conf[i][0] if y_pred[i] == 1 else 1 - conf[i][0]
                        conf_str = round(conf_str, 2)
                    else:
                        conf_str = round(conf[i][y_pred[i]], 2)
                    cv2.putText(src_img, text=str(conf_str), org=(30, 30), fontFace=cv2.FONT_ITALIC,
                                fontScale=0.7, color=(0, 0, 255), thickness=2)
                    save_img_file(dest, src_img)
                else:
                    if print_err:
                        print(file)
                    shutil.copyfile(file, dest)
            else:
                # print err
                print('perd ERR: %s: pred:%s  lable:%s' % (file, y_pred[i], y_true[i]))

    print('extract {} files to {} finished'.format(move_files, target_fold))


meanteacher_defaults_para = {
    # Technical details
    'workers': 2,
    'checkpoint_epochs': 1,  # default
    'evaluation-epochs': 1,  # default
    # Architecture
    'arch': 'resnet_50',
    'print_freq': 200,  # 未用，改为训练打印5次，验证打印2次

    # Costs
    'consistency_type': 'mse',  # default
    'consistency_rampup': 5,
    'consistency': 4.0,
    # 'logit_distance_cost': 0.01,
    'weight_decay': 2e-4,

    # Optimization
    'lr_rampup': 0,
    'base_lr': 0.001,
    'nesterov': True,
    'pretrained': True,

    'epochs': 150,
    'lr_rampdown_epochs': 400,
    'ema_decay': 0.99,  # default
}


def parameters(data_root, bs, patience=5, rate_step=[]):
    if len(rate_step) == 0:  # full
        data_seed = 100
        yield {
            **meanteacher_defaults_para,
            'patience': patience,
            'base_batch_size': bs,
            'base_labeled_batch_size': 0 if data_seed == 100 else int(bs / 2),
            'exclude-unlabeled': True if data_seed == 100 else False,
            'train_subdir': '%s' % os.path.join(data_root, 'train.txt'),
            'eval_subdir': '%s' % os.path.join(data_root, 'val.txt'),
            'labels': os.path.join(data_root, 'train.txt'),
        }
    else:
        assert len(rate_step) == 3
        start = rate_step[0]
        end = rate_step[1]
        step = rate_step[2]
        for data_seed in range(start, end, step):
            yield {
                **meanteacher_defaults_para,
                'patience': patience,
                'base_batch_size': bs,
                'base_labeled_batch_size': 0 if data_seed == 100 else int(bs / 2),
                'exclude-unlabeled': True if data_seed == 100 else False,
                'train_subdir': '%s' % os.path.join(data_root, 'train_%s.txt' % data_seed),
                'eval_subdir': '%s' % os.path.join(data_root, 'val_%s.txt' % data_seed),
                'labels': os.path.join(data_root, 'label_%s.txt' % data_seed),
            }


class Trainer(object):
    def __init__(self, n_labels, channel_stats, proj_name=None, data_root=None, result_path=None, bs=16, patience=5):
        self.train_transf = data.TransformTwice(transforms.Compose([
            data.RandomTranslateWithReflect(4),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ]))
        self.eval_transf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**channel_stats)
        ])

        self.parameters = parameters(data_root, bs, patience)
        self.arch = meanteacher_defaults_para['arch']
        self.n_labels = n_labels
        self.result_path = result_path
        self.proj_name = proj_name
        self.model_name = None
        self.model = None
        self.ema_model = None

    def _train(self, base_batch_size, base_labeled_batch_size, base_lr, **kwargs):
        ngpu = torch.cuda.device_count()
        assert ngpu > 0, "Expecting at least one GPU, found none."

        adapted_args = {
            'batch_size': base_batch_size * ngpu,
            'labeled_batch_size': base_labeled_batch_size * ngpu,
            'lr': base_lr * ngpu,
        }
        mean_teacher_main.args = parse_dict_args(**adapted_args, **kwargs)
        print('args:', mean_teacher_main.args)
        self.model_name = "{}_c{}_{}".format(self.proj_name, self.n_labels, get_time_stamp())
        self.model_name_base = self.model_name + '_base'
        self.model_name_ema = self.model_name + '_ema'
        context = RunContext(self.result_path, self.model_name)

        data_transformation = {
            'train_transformation': self.train_transf,
            'eval_transformation': self.eval_transf,
            'datadir': '',  # 未用
            'num_classes': self.n_labels
        }
        mean_teacher_main.main(context, data_transformation)

    def train(self):
        for run_params in self.parameters:
            self._train(**run_params)

    def load_best_model(self):
        self.model = mean_teacher_main.create_model(self.n_labels, self.arch,
                                                    checkpoint=os.path.join(self.result_path,
                                                                            self.model_name_base + '.pth'))
        self.ema_model = mean_teacher_main.create_model(self.n_labels, self.arch, ema=True,
                                                        checkpoint=os.path.join(self.result_path,
                                                                                self.model_name_ema + '.pth'))

    def load_model(self, base_model_file, ema_model_file):
        if base_model_file is not None:
            self.model = mean_teacher_main.create_model(self.n_labels, self.arch,
                                                        checkpoint=base_model_file)
            self.model_name_base = os.path.basename(base_model_file).split('_base')[0] + '_base'
        if ema_model_file is not None:
            self.ema_model = mean_teacher_main.create_model(self.n_labels, self.arch, ema=True,
                                                            checkpoint=ema_model_file)
            self.model_name_ema = os.path.basename(ema_model_file).split('_ema')[0] + '_ema'

    def predict_txtfile(self, model, txtfile, extract_all=False, add_conf=False, is_ema=False, binary_threshold=0.5):
        model.eval()  # 预测模式

        total_pred = []
        total_true = np.array([])
        total_zxd = []

        rst_str = '%s_%s' % (get_time_stamp(), self.model_name_ema  if is_ema else self.model_name_base)
        txtfile_name = os.path.basename(txtfile)[:-4]
        to_check_path_result = os.path.dirname(txtfile) + '_' + txtfile_name + '_result_%s' % rst_str
        if not os.path.exists(to_check_path_result):
            os.makedirs(to_check_path_result)

        eval_loader = torch.utils.data.DataLoader(
            mean_teacher_main.TextFileDataset(txtfile, self.eval_transf),
            batch_size=4,
            shuffle=False,
            num_workers=2 * 2,  # Needs images twice as fast
            pin_memory=True,
            drop_last=False)

        file_test = []
        predictions = []
        with torch.no_grad():
            for X, y, fn in eval_loader:
                X = X.cuda()
                y = y.cuda(non_blocking=True)
                # Prediction
                score = model(X)

                # _, prediction = torch.max(score, 1)
                # percentage = torch.nn.functional.softmax(score, dim=1)# * 100
                # percentage_list 一个bs预测列表，每个列表2个置信度，分别对应2个分类
                # percentage_list = percentage.cpu().detach().numpy().tolist()


                percentage_list = score.cpu().detach().numpy().tolist()

                if self.n_labels == 2 and binary_threshold != 0.5:
                    pred_cls = []
                    for item in percentage_list:
                        pred_cls.append(1 if item[1] > binary_threshold else 0)
                else:
                    pred_cls = [item.index(max(item)) for item in percentage_list]

                total_pred.extend(pred_cls)
                total_true = np.concatenate((total_true, y.data.cpu()))
                if self.n_labels == 2:
                    cls1_zxd = [item[1] for item in percentage_list]
                    total_zxd.extend(cls1_zxd)
                file_test.extend(fn)
                predictions.extend(percentage_list)

        Plot.show_matrix(total_pred, total_true, self.n_labels,
                         to_check_path_result + '/confusion_matrix_%s_' % rst_str, normalize=False)
        extract_pred_error_files(file_test, total_pred, total_true, to_check_path_result + '/pred_error',
                                 extract_all,
                                 predictions if add_conf else None, True)
        if self.n_labels == 2:
            auc = Plot.plot_roc(total_true, total_zxd, to_check_path_result + '/roc_%s_' % rst_str)

    def predict_single_file(self, model, file_path, binary_threshold=0.5, add_conf=False, print_rst=False):
        image_PIL = Image.open(file_path)
        #
        image_tensor = self.eval_transf(image_PIL)
        # 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
        image_tensor.unsqueeze_(0)
        image_tensor = image_tensor.cuda()
        out = model(image_tensor)
        percentage = out.cpu().detach().numpy().tolist()[0]

        if self.n_labels == 2 and binary_threshold != 0.5:
            y_pred = 1 if percentage[1] > binary_threshold else 0
        else:
            y_pred = np.argmax(percentage)

        conf = round(percentage[y_pred], 2)

        src_img = cv2.cvtColor(np.asarray(image_PIL), cv2.COLOR_RGB2BGR)
        if add_conf:
            cv2.putText(src_img, text=str(conf), org=(30, 30), fontFace=cv2.FONT_ITALIC,
                        fontScale=0.7, color=(0, 0, 255), thickness=2)
        if print_rst:
            print(file_path, y_pred, conf)
        return y_pred, conf, src_img

    def predict_single_path(self, files_path, target_path=None, binary_threshold=0.5, add_conf=False):
        files = fetch_all_files(files_path)
        for abs_file in files:
            y_pred, conf, src_img = self.predict_single_file(abs_file, binary_threshold, add_conf)
            if target_path:
                cls_target_path = os.path.join(target_path, str(y_pred[0]))
                os.makedirs(cls_target_path, exist_ok=True)
                save_img_file(os.path.join(cls_target_path, os.path.basename(abs_file)), src_img)
