# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt
from ..utils import load_variables
from ..files import get_file_list
from ..calc import mean_filter_padded
from ..models.utils import confusion_to_all, confusions_to_TFPN
from matplotlib.widgets import Slider, RadioButtons


#   交互式地显示FP和FN的变化情况

class Confusion_Visualization():
    def __init__(self, train_confusion_path, valid_confusion_path, class_names=None):

        self.plot_data = {}

        #   载入数据
        self.confusion_train = self._load_data(train_confusion_path)
        self.tfpn_train = confusions_to_TFPN(self.confusion_train)
        self.confusion_valid = self._load_data(valid_confusion_path)
        self.tfpn_valid = confusions_to_TFPN(self.confusion_valid)

        from pinglib.utils import save_variables
        save_variables(self.confusion_train,'confusions_train.pkl')
        save_variables(self.confusion_valid, 'confusions_validate.pkl')

        if self.confusion_train.shape[-1] != self.confusion_valid.shape[-1]:
            print('Training and Validation data not match (of length {} and {} respectively)'.
                  format(self.confusion_train.shape[-1], self.confusion_valid.shape[-1]))
            raise ValueError
        else:
            self.series_length = self.confusion_train.shape[-1]
            self.class_amount = self.confusion_train.shape[0]
        if class_names is None:
            self.class_names = [str(i) for i in range(self.class_amount)]
        elif isinstance(class_names, list) or isinstance(class_names, tuple):
            if len(class_names) == self.class_amount:
                self.class_names = class_names
            else:
                print('Class names ({}) mismatch with confusion ({})'.format(len(class_names), self.class_amount))
                raise ValueError
        else:
            print('Class names unsupported (please use list, e.g. [''Benign'',''Tumor''])')
            raise ValueError

        #   分析数据
        self.performance_train = self._process_data(self.confusion_train)
        self.performance_valid = self._process_data(self.confusion_valid)

        #   绘制框架
        self._draw_framework()

        #   绘制控制面板
        self._draw_control_panel()

        #   绘制初始图像
        self._figure_initialize()

    def _draw_framework(self):
        self.heatmap_fig = plt.figure(1, figsize=(8, 4))  # 用于画热图
        plt.title('Heatmap')
        self.heatmap_train = plt.subplot(1, 2, 1)
        self.heatmap_train.set_title('Train')
        self.heatmap_train.set_xticks(np.arange(self.class_amount))
        self.heatmap_train.set_yticks(np.arange(self.class_amount))
        self.heatmap_train.set_xticklabels(self.class_names)
        self.heatmap_train.set_yticklabels(self.class_names)
        self.heatmap_train.set_ylabel('Ground Truth')
        self.heatmap_train.set_xlabel('Prediction')
        self.heatmap_valid = plt.subplot(1, 2, 2)
        self.heatmap_valid.set_title('Validation')
        self.heatmap_valid.set_xticks(np.arange(self.class_amount))
        self.heatmap_valid.set_yticks(np.arange(self.class_amount))
        self.heatmap_valid.set_xticklabels(self.class_names)
        self.heatmap_valid.set_yticklabels(self.class_names)
        self.heatmap_valid.set_ylabel('Ground Truth')
        self.heatmap_valid.set_xlabel('Prediction')

        self.performance_fig = plt.figure(2, figsize=(10, 6))  # 用于画性能图
        plt.title('Performance')
        self.performance_global_p_r = plt.subplot(2, 2, 1)
        self.performance_global_p_r.set_title('Global')
        self.performance_class_p_r = plt.subplot(2, 2, 2)
        self.performance_class_p_r.set_title('Class')
        self.performance_global_a_f = plt.subplot(2, 2, 3)
        self.performance_class_f = plt.subplot(2, 2, 4)

        self.trace_fig = plt.figure(3, figsize=(6, 6))
        self.trace_plot = plt.subplot(1, 1, 1)
        self.trace_plot.set_xlabel('FP')
        self.trace_plot.set_ylabel('FN')

        self.controller = plt.figure(4, figsize=(4, 4))

    def _draw_control_panel(self):

        self.ctrl_cursor = 0
        self.ctrl_class = 0
        self.ctrl_smooth = 1
        self.ctrl_trajectory = 1
        self.ctrl_normalization = 'None'

        ax = self.controller.add_axes([0.2, 0.90, 0.5, 0.05])
        self.slider_cursor = DiscreteSlider(label='cursor', valmin=0, valmax=self.series_length - 1,
                                            ax=ax, increment=1, valinit=self.ctrl_cursor)
        self.slider_cursor.on_changed(self.controller_change_slider_cursor)

        ax = self.controller.add_axes([0.2, 0.70, 0.5, 0.05])
        self.slider_class = DiscreteSlider(label='class', valmin=0, valmax=self.class_amount,
                                           ax=ax, increment=1, valinit=self.ctrl_class)
        self.slider_class.on_changed(self.controller_change_slider_class)

        ax = self.controller.add_axes([0.2, 0.50, 0.5, 0.05])
        self.slider_smooth = DiscreteSlider(label='smooth', valmin=1, valmax=self.series_length,
                                            ax=ax, increment=1, valinit=self.ctrl_smooth)
        self.slider_smooth.on_changed(self.controller_change_slider_smooth)

        ax = self.controller.add_axes([0.2, 0.30, 0.5, 0.05])
        self.slider_trajectory = DiscreteSlider(label='trajectory', valmin=1, valmax=self.series_length,
                                                ax=ax, increment=1, valinit=self.ctrl_trajectory)
        self.slider_trajectory.on_changed(self.controller_change_slider_trajectory)

        ax = self.controller.add_axes([0.6, 0.1, 0.15, 0.15])
        self.radiobuttons_norm = RadioButtons(ax, ('None', 'TP', 'TP+TN', 'All'), active=self.ctrl_normalization)
        self.radiobuttons_norm.on_clicked(self.controller_change_radiobuttons_norm)

    def _load_data(self, obj):
        if os.path.isfile(obj):
            #   这时候直接读变量就行了
            [conf_tensor] = load_variables(obj)
            return conf_tensor
        elif os.path.isdir(obj):
            #   这时候需要逐个读文件
            file_list = get_file_list(obj)
            file_amount = len(file_list)
            conf_list = []
            for i in range(file_amount):
                [conf] = load_variables(os.path.join(obj, str(i) + '.pkl'))
                conf_list.append(conf)
            return np.concatenate(np.expand_dims(conf_list, axis=-1), axis=-1)
        elif isinstance(obj, np.ndarray):
            return obj
        else:
            print('Invalid confusion source: {}'.format(obj))
            raise ValueError

    #   confusions: c x c x N
    def _process_data(self, confusions):
        performance = {}
        performance['accuracy'] = []
        performance['precision'] = []
        performance['recall'] = []
        performance['f1'] = []
        for i in range(confusions.shape[-1]):
            current_result = confusion_to_all(confusions[:, :, i])
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                performance[metric].append(current_result[metric])
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            performance[metric] = np.concatenate(np.expand_dims(performance[metric], axis=-1), axis=-1)
        for metric in ['precision', 'recall', 'f1']:
            global_performance = np.mean(performance[metric], axis=0)
            global_performance[np.isnan(global_performance)] = 0
            performance['global_' + metric] = global_performance
        #   全局性能处理完nan之后再处理分类性能
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            performance[metric][np.isnan(performance[metric])] = 0
        performance['global_accuracy'] = performance['accuracy']

        return performance

    #   这里会对一些图表元素进行初始化
    def _figure_initialize(self):

        self.element_heatmap_train_text_pool = []  # 这两个用于存储heatmap上的文字
        self.element_heatmap_valid_text_pool = []
        self._update_figure_heatmap()

        self.element_performance_global_precision_train = None
        self.element_performance_global_recall_train = None
        self.element_performance_global_precision_valid = None
        self.element_performance_global_recall_valid = None
        self.element_performance_global_accuracy_train = None
        self.element_performance_global_f1_train = None
        self.element_performance_global_accuracy_valid = None
        self.element_performance_global_f1_valid = None
        self._update_figure_global_performance()

        self.element_performance_class_precision_train = None
        self.element_performance_class_recall_train = None
        self.element_performance_class_precision_valid = None
        self.element_performance_class_recall_valid = None
        self.element_performance_class_f1_train = None
        self.element_performance_class_f1_valid = None
        self._update_figure_class_performance()

        self.element_trajectory_train = None
        self.element_trajectory_valid = None
        self._update_figure_trajectory()

        plt.show()

    def controller_change_slider_cursor(self, event):
        new_value = int(self.slider_cursor.val)
        if self.ctrl_cursor != new_value:
            self.ctrl_cursor = new_value
            self._update_figure_heatmap()
            self._update_figure_trajectory()

    def controller_change_slider_class(self, event):
        new_value = int(self.slider_class.val)
        if self.ctrl_class != new_value:
            self.ctrl_class = new_value
            self._update_figure_class_performance()
            self._update_figure_trajectory()
        pass

    def controller_change_slider_smooth(self, event):
        new_value = int(self.slider_smooth.val)
        if self.ctrl_smooth != new_value:
            self.ctrl_smooth = new_value
            self._update_figure_global_performance()
            self._update_figure_class_performance()
        pass

    def controller_change_slider_trajectory(self, event):
        new_value = int(self.slider_trajectory.val)
        if self.ctrl_trajectory != new_value:
            self.ctrl_trajectory = new_value
            self._update_figure_trajectory()
        pass

    def controller_change_radiobuttons_norm(self, event):
        new_value = self.radiobuttons_norm.value_selected
        if self.ctrl_normalization != new_value:
            self.ctrl_normalization = new_value
            self._update_figure_trajectory()
        pass

    def _update_data_heatmap(self):
        ctrl_cursor = self.ctrl_cursor
        self.plot_data['heatmap_train'] = self.confusion_train[:, :, ctrl_cursor] / np.sum(
            self.confusion_train[:, :, ctrl_cursor], axis=1, keepdims=True)
        self.plot_data['heatmap_valid'] = self.confusion_valid[:, :, ctrl_cursor] / np.sum(
            self.confusion_valid[:, :, ctrl_cursor], axis=1, keepdims=True)

    def _update_data_global_performance(self):
        ctrl_smooth = self.ctrl_smooth
        for metric in ['precision', 'recall', 'f1', 'accuracy']:
            self.plot_data['performance_global_train_' + metric] = mean_filter_padded(
                self.performance_train['global_' + metric], ctrl_smooth)
            self.plot_data['performance_global_valid_' + metric] = mean_filter_padded(
                self.performance_valid['global_' + metric], ctrl_smooth)

    def _update_data_class_performance(self):
        ctrl_class = self.ctrl_class
        ctrl_smooth = self.ctrl_smooth
        for metric in ['precision', 'recall', 'f1']:
            self.plot_data['performance_class_train_' + metric] = mean_filter_padded(
                self.performance_train[metric][ctrl_class, :], ctrl_smooth)
            self.plot_data['performance_class_valid_' + metric] = mean_filter_padded(
                self.performance_valid[metric][ctrl_class, :], ctrl_smooth)

    def _update_data_trajectory(self):
        ctrl_cursor = self.ctrl_cursor
        ctrl_class = self.ctrl_class
        ctrl_trajectory = self.ctrl_trajectory
        ctrl_normalization = self.ctrl_normalization

        #   希望摘选的范围是，从指定点开始，向后推trajectory步！
        lb = max(ctrl_cursor, 0)
        ub = min(ctrl_cursor + ctrl_trajectory, self.series_length)

        train_tp = self.tfpn_train[ctrl_class, 0, lb:ub]
        train_tn = self.tfpn_train[ctrl_class, 1, lb:ub]
        train_fp = self.tfpn_train[ctrl_class, 2, lb:ub]
        train_fn = self.tfpn_train[ctrl_class, 3, lb:ub]

        valid_tp = self.tfpn_valid[ctrl_class, 0, lb:ub]
        valid_tn = self.tfpn_valid[ctrl_class, 1, lb:ub]
        valid_fp = self.tfpn_valid[ctrl_class, 2, lb:ub]
        valid_fn = self.tfpn_valid[ctrl_class, 3, lb:ub]

        if ctrl_normalization == 'None':
            #   None的情况
            self.plot_data['train_fp'] = train_fp
            self.plot_data['train_fn'] = train_fn
            self.plot_data['valid_fp'] = valid_fp
            self.plot_data['valid_fn'] = valid_fn
        elif ctrl_normalization == 'TP':
            #   TP
            self.plot_data['train_fp'] = train_fp / train_tp
            self.plot_data['train_fn'] = train_fn / train_tp
            self.plot_data['valid_fp'] = valid_fp / valid_tp
            self.plot_data['valid_fn'] = valid_fn / valid_tp
        elif ctrl_normalization == 'TP+TN':
            #   TP+TN
            self.plot_data['train_fp'] = train_fp / (train_tp + train_tn)
            self.plot_data['train_fn'] = train_fn / (train_tp + train_tn)
            self.plot_data['valid_fp'] = valid_fp / (valid_tp + valid_tn)
            self.plot_data['valid_fn'] = valid_fn / (valid_tp + valid_tn)
        elif ctrl_normalization == 'All':
            #   All
            self.plot_data['train_fp'] = train_fp / (train_tp + train_tn + train_fp + train_fn)
            self.plot_data['train_fn'] = train_fn / (train_tp + train_tn + train_fp + train_fn)
            self.plot_data['valid_fp'] = valid_fp / (valid_tp + valid_tn + valid_fp + valid_fn)
            self.plot_data['valid_fn'] = valid_fn / (valid_tp + valid_tn + valid_fp + valid_fn)
        else:
            raise ValueError

    def _update_figure_heatmap(self):
        #   更新数据
        self._update_data_heatmap()

        #   绘制热图
        heatmap_data_train = self.plot_data['heatmap_train']
        heatmap_data_valid = self.plot_data['heatmap_valid']

        #   先清除之前写的字
        for item in self.element_heatmap_train_text_pool:
            item.remove()
        for item in self.element_heatmap_valid_text_pool:
            item.remove()
        self.element_heatmap_train_text_pool = []
        self.element_heatmap_valid_text_pool = []

        #   然后写新的字
        for i in range(self.class_amount):
            for j in range(self.class_amount):
                obj = self.heatmap_train.text(j, i, '{:.2f}'.format(heatmap_data_train[i, j]),
                                              ha="center", va="center", color="w",
                                              fontsize=8)
                self.element_heatmap_train_text_pool.append(obj)
        self.heatmap_train.imshow(heatmap_data_train, cmap='jet')

        for i in range(self.class_amount):
            for j in range(self.class_amount):
                obj = self.heatmap_valid.text(j, i, '{:.2f}'.format(heatmap_data_train[i, j]),
                                              ha="center", va="center", color="w",
                                              fontsize=8)
                self.element_heatmap_valid_text_pool.append(obj)
        self.heatmap_valid.imshow(heatmap_data_valid, cmap='jet')

        self.heatmap_fig.tight_layout()

        self.heatmap_fig.canvas.draw_idle()

    def _update_figure_global_performance(self):
        time_steps = np.arange(self.series_length)
        self._update_data_global_performance()

        if self.element_performance_global_precision_train is None:
            self.element_performance_global_precision_train, = \
                self.performance_global_p_r.plot(time_steps,
                                                 self.plot_data['performance_global_train_precision'],
                                                 color='red', linestyle='-')
        else:
            self.element_performance_global_precision_train.set_ydata(
                self.plot_data['performance_global_train_precision'])

        if self.element_performance_global_recall_train is None:
            self.element_performance_global_recall_train, = \
                self.performance_global_p_r.plot(time_steps,
                                                 self.plot_data['performance_global_train_recall'],
                                                 color='blue', linestyle='-')
        else:
            self.element_performance_global_recall_train.set_ydata(self.plot_data['performance_global_train_recall'])

        if self.element_performance_global_precision_valid is None:
            self.element_performance_global_precision_valid, = \
                self.performance_global_p_r.plot(time_steps,
                                                 self.plot_data['performance_global_valid_precision'],
                                                 color='red', linestyle='--')
        else:
            self.element_performance_global_precision_valid.set_ydata(
                self.plot_data['performance_global_valid_precision'])

        if self.element_performance_global_recall_valid is None:
            self.element_performance_global_recall_valid, = \
                self.performance_global_p_r.plot(time_steps,
                                                 self.plot_data['performance_global_valid_recall'],
                                                 color='blue', linestyle='--')
            #   图例，放在第一次画图的时候加
            self.performance_global_p_r.legend(
                ['pre_train', 'rec_train', 'pre_valid', 'rec_valid'])
        else:
            self.element_performance_global_recall_valid.set_ydata(self.plot_data['performance_global_valid_recall'])

        if self.element_performance_global_accuracy_train is None:
            self.element_performance_global_accuracy_train, = \
                self.performance_global_a_f.plot(time_steps,
                                                 self.plot_data['performance_global_train_accuracy'],
                                                 color='red', linestyle='-')
        else:
            self.element_performance_global_accuracy_train.set_ydata(
                self.plot_data['performance_global_train_accuracy'])

        if self.element_performance_global_f1_train is None:
            self.element_performance_global_f1_train, = \
                self.performance_global_a_f.plot(time_steps,
                                                 self.plot_data['performance_global_train_f1'],
                                                 color='blue', linestyle='-')
        else:
            self.element_performance_global_f1_train.set_ydata(self.plot_data['performance_global_train_f1'])

        if self.element_performance_global_accuracy_valid is None:
            self.element_performance_global_accuracy_valid, = \
                self.performance_global_a_f.plot(time_steps,
                                                 self.plot_data['performance_global_valid_accuracy'],
                                                 color='red', linestyle='--')
        else:
            self.element_performance_global_accuracy_valid.set_ydata(
                self.plot_data['performance_global_valid_accuracy'])

        if self.element_performance_global_f1_valid is None:
            self.element_performance_global_f1_valid, = \
                self.performance_global_a_f.plot(time_steps,
                                                 self.plot_data['performance_global_valid_f1'],
                                                 color='blue', linestyle='--')
            #   图例，放在第一次画图的时候加
            self.performance_global_a_f.legend(
                ['acc_train', 'f1_train', 'acc_valid', 'f1_valid'])
        else:
            self.element_performance_global_f1_valid.set_ydata(self.plot_data['performance_global_valid_f1'])

        self.performance_fig.canvas.draw_idle()

    def _update_figure_class_performance(self):

        time_steps = np.arange(self.series_length)
        self._update_data_class_performance()

        if self.element_performance_class_precision_train is None:
            self.element_performance_class_precision_train, = \
                self.performance_class_p_r.plot(time_steps,
                                                self.plot_data['performance_class_train_precision'],
                                                color='red', linestyle='-')
        else:
            self.element_performance_class_precision_train.set_ydata(
                self.plot_data['performance_global_train_precision'])

        if self.element_performance_class_recall_train is None:
            self.element_performance_class_recall_train, = \
                self.performance_class_p_r.plot(time_steps,
                                                self.plot_data['performance_class_train_recall'],
                                                color='blue', linestyle='-')
        else:
            self.element_performance_class_recall_train.set_ydata(self.plot_data['performance_class_train_recall'])

        if self.element_performance_class_precision_valid is None:
            self.element_performance_class_precision_valid, = \
                self.performance_class_p_r.plot(time_steps,
                                                self.plot_data['performance_class_valid_precision'],
                                                color='red', linestyle='--')
        else:
            self.element_performance_class_precision_valid.set_ydata(
                self.plot_data['performance_class_valid_precision'])

        if self.element_performance_class_recall_valid is None:
            self.element_performance_class_recall_valid, = \
                self.performance_class_p_r.plot(time_steps,
                                                self.plot_data['performance_class_valid_recall'],
                                                color='blue', linestyle='--')
            #   图例，放在第一次画图的时候加
            self.performance_class_p_r.legend(
                ['pre_train', 'rec_train', 'pre_valid', 'rec_valid'])
        else:
            self.element_performance_class_recall_valid.set_ydata(self.plot_data['performance_class_valid_recall'])

        if self.element_performance_class_f1_train is None:
            self.element_performance_class_f1_train, = \
                self.performance_class_f.plot(time_steps,
                                              self.plot_data['performance_class_train_f1'],
                                              color='blue', linestyle='-')
        else:
            self.element_performance_class_f1_train.set_ydata(self.plot_data['performance_class_train_f1'])

        if self.element_performance_class_f1_valid is None:
            self.element_performance_class_f1_valid, = \
                self.performance_class_f.plot(time_steps,
                                              self.plot_data['performance_class_valid_f1'],
                                              color='blue', linestyle='--')
            #   图例，放在第一次画图的时候加
            self.performance_class_f.legend(
                ['f1_train', 'f1_valid'])
        else:
            self.element_performance_class_f1_valid.set_ydata(self.plot_data['performance_class_valid_f1'])

        self.performance_fig.canvas.draw_idle()

    def _update_figure_trajectory(self):
        self._update_data_trajectory()

        data_length = len(self.plot_data['train_fp'])
        interpolation_values = (np.arange(data_length) + 1) / data_length
        blue_color = color_interpolation(np.array((1, 1, 1)), np.array((0, 0, 1)), interpolation_values)
        red_color = color_interpolation(np.array((1, 1, 1)), np.array((1, 0, 0)), interpolation_values)

        #   在这里对颜色矩阵进行了逆序，目的是为了让纯蓝和纯红显示在图例上（而不是显示白色）
        #   还需要小心数据长度为1的情况
        if blue_color.shape[0] > 1:
            blue_color = blue_color[::-1, :]
        if red_color.shape[0] > 1:
            red_color = red_color[::-1, :]

        if self.element_trajectory_train is not None:
            self.element_trajectory_train.remove()
        self.element_trajectory_train = self.trace_plot.scatter(self.plot_data['train_fp'][::-1],
                                                                self.plot_data['train_fn'][::-1],
                                                                c=blue_color)

        if self.element_trajectory_valid is not None:
            self.element_trajectory_valid.remove()
        self.element_trajectory_valid = self.trace_plot.scatter(self.plot_data['valid_fp'][::-1],
                                                                self.plot_data['valid_fn'][::-1],
                                                                c=red_color)

        max_range = np.max(np.concatenate((self.plot_data['train_fp'], self.plot_data['train_fn'],
                                           self.plot_data['valid_fp'], self.plot_data['valid_fn'])))
        self.trace_plot.set_xlim((0, 1.1 * max_range))
        self.trace_plot.set_ylim((0, 1.1 * max_range))
        self.trace_plot.set_aspect(1)

        self.trace_plot.legend(['Train', 'Valid'])

        self.trace_fig.canvas.draw_idle()


#   Copied from https://stackoverflow.com/questions/13656387/can-i-make-matplotlib-sliders-more-discrete
class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""

    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "increment" kwarg.
        "increment" specifies the step size that the slider will be discritized
        to."""
        self.inc = kwargs.pop('increment', 0.5)
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        discrete_val = int(val / self.inc) * self.inc
        # We can't just call Slider.set_val(self, discrete_val), because this
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon:
            self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson:
            return
        if len(self.observers.keys()) == 0:
            return
        for cid, func in self.observers.items():
            func(discrete_val)


#   用于获得渐变的颜色
#   values是从0~1的数，为一个一维数组，0对应color1，1对应color2
def color_interpolation(color_1, color_2, values):
    color_1 = np.reshape(color_1, (1, 3))
    color_2 = np.reshape(color_2, (1, 3))
    values = np.reshape(values, (-1, 1))
    return color_1 * (1 - values) + color_2 * values
