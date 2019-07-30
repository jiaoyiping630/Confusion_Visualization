# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from matplotlib import pyplot as plt
from ..utils import load_variables
from ..files import get_file_list
from ..calc import mean_filter_padded
from ..models.utils import confusion_to_all, confusions_to_TFPN
from matplotlib.widgets import Button, Slider, RadioButtons, CheckButtons


class Confusion_Visualizer():
    def __init__(self, train_confusion_path, valid_confusion_path, class_names=None):

        #   Load data
        self.confusion_train = self._load_data(train_confusion_path)
        self.tfpn_train = confusions_to_TFPN(self.confusion_train)
        self.confusion_valid = self._load_data(valid_confusion_path)
        self.tfpn_valid = confusions_to_TFPN(self.confusion_valid)

        #   Check whether dimensions match
        if self.confusion_train.shape[-1] != self.confusion_valid.shape[-1]:
            min_length = min(self.confusion_train.shape[-1], self.confusion_valid.shape[-1])
            print('Training and Validation data not match (of shape {} and {} respectively), truncated as {}'.
                  format(self.confusion_train.shape, self.confusion_valid.shape, min_length))
            self.confusion_train = self.confusion_train[:, :, 0:min_length]
            self.confusion_valid = self.confusion_valid[:, :, 0:min_length]
        self.series_length = self.confusion_train.shape[-1]
        self.class_amount = self.confusion_train.shape[0]

        #   Obtain class labels
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

        #   Process data to get performance
        self.plot_data = {}
        self.performance_train = self._process_data(self.confusion_train)
        self.performance_valid = self._process_data(self.confusion_valid)

        #   Set initial value for control panel (used in framework)
        self.ctrl_cursor = 0
        self.ctrl_class = 0
        self.ctrl_smooth = 1
        self.ctrl_trajectory_length = 1
        self.ctrl_trajectory_mode = 'FP-FN'
        self.ctrl_hold_range = False
        self.ctrl_same_ratio = True

        #   Draw framework
        self._draw_framework()

        #   Draw control panel
        self._draw_control_panel()

        #   Draw initial figure
        self._figure_initialize()

    def _draw_framework(self):

        #   Used for drawing confusion matrix (heatmap)
        self.heatmap_fig = plt.figure(1, figsize=(8, 4))
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

        #   Used for drawing performance
        self.performance_fig = plt.figure(2, figsize=(10, 6))
        plt.title('Performance')
        self.performance_global_p_r = plt.subplot(2, 2, 1)
        self.performance_global_p_r.set_title('Global')
        self.performance_class_p_r = plt.subplot(2, 2, 2)
        self.performance_class_p_r.set_title('Class: ' + self.class_names[int(self.ctrl_class)])
        self.performance_global_a_f = plt.subplot(2, 2, 3)
        self.performance_class_f = plt.subplot(2, 2, 4)

        #   Used for drawing FP/FN, etc.
        self.trace_fig = plt.figure(3, figsize=(6, 6))
        self.trace_plot = plt.subplot(1, 1, 1)
        self.trace_plot.set_xlabel('FP')
        self.trace_plot.set_ylabel('FN')
        self.trace_plot.set_title('Class: ' + self.class_names[int(self.ctrl_class)])

        #   Used for drawing control panel
        self.controller = plt.figure(4, figsize=(4, 4))

    def _draw_control_panel(self):

        ax = self.controller.add_axes([0.2, 0.90, 0.5, 0.05])
        self.slider_cursor = DiscreteSlider(label='cursor', valmin=0, valmax=self.series_length - 1,
                                            ax=ax, increment=1, valinit=self.ctrl_cursor)
        self.slider_cursor.on_changed(self.controller_change_slider_cursor)

        ax = self.controller.add_axes([0.2, 0.75, 0.5, 0.05])
        self.slider_class = DiscreteSlider(label='class', valmin=0, valmax=self.class_amount - 1,
                                           ax=ax, increment=1, valinit=self.ctrl_class)
        self.slider_class.on_changed(self.controller_change_slider_class)

        ax = self.controller.add_axes([0.2, 0.60, 0.5, 0.05])
        self.slider_smooth = DiscreteSlider(label='smooth', valmin=1, valmax=self.series_length,
                                            ax=ax, increment=1, valinit=self.ctrl_smooth)
        self.slider_smooth.on_changed(self.controller_change_slider_smooth)

        ax = self.controller.add_axes([0.2, 0.45, 0.5, 0.05])
        self.slider_trajectory = DiscreteSlider(label='trajectory', valmin=1, valmax=self.series_length,
                                                ax=ax, increment=1, valinit=self.ctrl_trajectory_length)
        self.slider_trajectory.on_changed(self.controller_change_slider_trajectory)

        ax = self.controller.add_axes([0.6, 0.1, 0.3, 0.25])
        self.radiobuttons_norm = RadioButtons(ax, ('FP-FN', 'FPR-TPR', 'Recl-Prec', 'Spec-Sens'), active=0)
        self.radiobuttons_norm.on_clicked(self.controller_change_radiobuttons_norm)

        ax = self.controller.add_axes([0.3, 0.1, 0.3, 0.25])
        self.checkbutton_hold = CheckButtons(ax, labels=['Hold on Range', 'Same Ratio'], actives=[False, True])
        self.checkbutton_hold.on_clicked(self.controller_change_checkbutton_hold)

    def _load_data(self, obj):
        if os.path.isfile(obj):
            #   read directly given a single file
            [conf_tensor] = load_variables(obj)
            return conf_tensor
        elif os.path.isdir(obj):
            #   read 0.pkl ~ *.pkl given a directory path
            file_list = get_file_list(obj)
            file_amount = len(file_list)
            conf_list = []
            for i in range(file_amount):
                [conf] = load_variables(os.path.join(obj, str(i) + '.pkl'))
                conf_list.append(conf)
            return np.concatenate(np.expand_dims(conf_list, axis=-1), axis=-1)
        elif isinstance(obj, np.ndarray):
            #   return directly given np array
            return obj
        else:
            print('Invalid confusion source: {}'.format(obj))
            raise ValueError

    #   Confusions: c x c x N
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
        #   If some performance is nan, it will replaced by 0
        for metric in ['precision', 'recall', 'f1']:
            global_performance = np.mean(performance[metric], axis=0)
            global_performance[np.isnan(global_performance)] = 0
            performance['global_' + metric] = global_performance
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            performance[metric][np.isnan(performance[metric])] = 0
        performance['global_accuracy'] = performance['accuracy']

        return performance

    #   Figure initialization
    def _figure_initialize(self):

        self.element_heatmap_train_text_pool = []
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

    #   Event for slider cursor changed
    def controller_change_slider_cursor(self, event):
        new_value = int(self.slider_cursor.val)
        if self.ctrl_cursor != new_value:
            self.ctrl_cursor = new_value
            self._update_figure_heatmap()
            self._update_figure_trajectory()

    #   Event for slider class changed
    def controller_change_slider_class(self, event):
        new_value = int(self.slider_class.val)
        if self.ctrl_class != new_value:
            self.ctrl_class = new_value
            self._update_figure_class_performance()
            self._update_figure_trajectory()

    #   Event for slider smooth changed
    def controller_change_slider_smooth(self, event):
        new_value = int(self.slider_smooth.val)
        if self.ctrl_smooth != new_value:
            self.ctrl_smooth = new_value
            self._update_figure_global_performance()
            self._update_figure_class_performance()

    #   Event for slider trajectory (length) changed
    def controller_change_slider_trajectory(self, event):
        new_value = int(self.slider_trajectory.val)
        if self.ctrl_trajectory_length != new_value:
            self.ctrl_trajectory_length = new_value
            self._update_figure_trajectory()

    #   Event for radiobuttons changed (FP/FN normalization method)
    def controller_change_radiobuttons_norm(self, event):
        new_value = self.radiobuttons_norm.value_selected
        if self.ctrl_trajectory_mode != new_value:
            self.ctrl_trajectory_mode = new_value
            self._update_figure_trajectory()

    #   Event for checkbox changed (hold FP/FN figure's range)
    def controller_change_checkbutton_hold(self, event):
        new_value = self.checkbutton_hold.get_status()
        self.ctrl_hold_range = new_value[0]
        self.ctrl_same_ratio = new_value[1]

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
        ctrl_trajectory = self.ctrl_trajectory_length
        ctrl_trajectory_mode = self.ctrl_trajectory_mode

        #   Desired range: from cursor to (cursor+trajectory)
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

        #   Measurements refer to: http://www.davidsbatista.net/blog/2018/08/19/NLP_Metrics/
        if ctrl_trajectory_mode == 'FP-FN':
            self.plot_data['trajectory_train_x'] = train_fp
            self.plot_data['trajectory_train_y'] = train_fn
            self.plot_data['trajectory_valid_x'] = valid_fp
            self.plot_data['trajectory_valid_y'] = valid_fn
        elif ctrl_trajectory_mode == 'FPR-TPR':
            self.plot_data['trajectory_train_x'] = train_fp / (train_tn + train_fp)
            self.plot_data['trajectory_train_y'] = train_tp / (train_tp + train_fn)
            self.plot_data['trajectory_valid_x'] = valid_fp / (valid_tn + valid_fp)
            self.plot_data['trajectory_valid_y'] = valid_tp / (valid_tp + valid_fn)
        elif ctrl_trajectory_mode == 'Recl-Prec':
            self.plot_data['trajectory_train_x'] = train_tp / (train_tp + train_fn)
            self.plot_data['trajectory_train_y'] = train_tp / (train_tp + train_fp)
            self.plot_data['trajectory_valid_x'] = valid_tp / (valid_tp + valid_fn)
            self.plot_data['trajectory_valid_y'] = valid_tp / (valid_tp + valid_fp)
        elif ctrl_trajectory_mode == 'Spec-Sens':
            self.plot_data['trajectory_train_x'] = train_tn / (train_tn + train_fp)
            self.plot_data['trajectory_train_y'] = train_tp / (train_tp + train_fn)
            self.plot_data['trajectory_valid_x'] = valid_tn / (valid_tn + valid_fp)
            self.plot_data['trajectory_valid_y'] = valid_tp / (valid_tp + valid_fn)
        else:
            raise ValueError
        pass

    def _update_figure_heatmap(self):

        #   Fetch data
        self._update_data_heatmap()
        heatmap_data_train = self.plot_data['heatmap_train']
        heatmap_data_valid = self.plot_data['heatmap_valid']

        #   Clean former text
        for item in self.element_heatmap_train_text_pool:
            item.remove()
        for item in self.element_heatmap_valid_text_pool:
            item.remove()
        self.element_heatmap_train_text_pool = []
        self.element_heatmap_valid_text_pool = []

        #   Write new text & Draw
        for i in range(self.class_amount):
            for j in range(self.class_amount):
                obj = self.heatmap_train.text(j, i, '{:.2f}'.format(heatmap_data_train[i, j]),
                                              ha="center", va="center", color="w",
                                              fontsize=8)
                self.element_heatmap_train_text_pool.append(obj)
        self.heatmap_train.imshow(heatmap_data_train, cmap='jet')

        for i in range(self.class_amount):
            for j in range(self.class_amount):
                obj = self.heatmap_valid.text(j, i, '{:.2f}'.format(heatmap_data_valid[i, j]),
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
            #   Legend, only added at first drawing
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
            #   Legend, only added at first drawing
            self.performance_global_a_f.legend(
                ['acc_train', 'f1_train', 'acc_valid', 'f1_valid'])
        else:
            self.element_performance_global_f1_valid.set_ydata(self.plot_data['performance_global_valid_f1'])

        self.performance_fig.canvas.draw_idle()

    def _update_figure_class_performance(self):

        time_steps = np.arange(self.series_length)
        self._update_data_class_performance()

        self.performance_class_p_r.set_title('Class: ' + self.class_names[int(self.ctrl_class)])

        if self.element_performance_class_precision_train is None:
            self.element_performance_class_precision_train, = \
                self.performance_class_p_r.plot(time_steps,
                                                self.plot_data['performance_class_train_precision'],
                                                color='red', linestyle='-')
        else:
            self.element_performance_class_precision_train.set_ydata(
                self.plot_data['performance_class_train_precision'])

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
            #   Legend, only added at first drawing
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
            #   Legend, only added at first drawing
            self.performance_class_f.legend(
                ['f1_train', 'f1_valid'])
        else:
            self.element_performance_class_f1_valid.set_ydata(self.plot_data['performance_class_valid_f1'])

        self.performance_fig.canvas.draw_idle()

    def _update_figure_trajectory(self):
        self._update_data_trajectory()

        self.trace_plot.set_title('Class: ' + self.class_names[int(self.ctrl_class)])

        data_length = len(self.plot_data['trajectory_train_x'])
        interpolation_values = (np.arange(data_length) + 1) / data_length
        blue_color = color_interpolation(np.array((1, 1, 1)), np.array((0, 0, 1)), interpolation_values)
        red_color = color_interpolation(np.array((1, 1, 1)), np.array((1, 0, 0)), interpolation_values)

        #   To make the pure blue and pure red showed on legend, we reverse the array
        if blue_color.shape[0] > 1:
            blue_color = blue_color[::-1, :]
        if red_color.shape[0] > 1:
            red_color = red_color[::-1, :]

        if self.element_trajectory_train is not None:
            self.element_trajectory_train.remove()
        self.element_trajectory_train = self.trace_plot.scatter(self.plot_data['trajectory_train_x'][::-1],
                                                                self.plot_data['trajectory_train_y'][::-1],
                                                                c=blue_color)

        if self.element_trajectory_valid is not None:
            self.element_trajectory_valid.remove()
        self.element_trajectory_valid = self.trace_plot.scatter(self.plot_data['trajectory_valid_x'][::-1],
                                                                self.plot_data['trajectory_valid_y'][::-1],
                                                                c=red_color)

        #   Adjust axis range
        if not self.ctrl_hold_range:
            max_range_x = np.max(
                np.concatenate((self.plot_data['trajectory_train_x'], self.plot_data['trajectory_valid_x'])))
            max_range_y = np.max(
                np.concatenate((self.plot_data['trajectory_train_y'], self.plot_data['trajectory_valid_y'])))
            max_range = np.max([max_range_x, max_range_y])

            if self.ctrl_same_ratio:
                self.trace_plot.set_aspect('equal')
                self.trace_plot.set_xlim((0, 1.1 * max_range))
                self.trace_plot.set_ylim((0, 1.1 * max_range))
            else:
                self.trace_plot.set_aspect('auto')
                self.trace_plot.set_xlim((0, 1.1 * max_range_x))
                self.trace_plot.set_ylim((0, 1.1 * max_range_y))

        #   Updata axis label
        if self.ctrl_trajectory_mode == 'FP-FN':
            self.trace_plot.set_xlabel('FP')
            self.trace_plot.set_ylabel('FN')
        elif self.ctrl_trajectory_mode == 'FPR-TPR':
            self.trace_plot.set_xlabel('FPR')
            self.trace_plot.set_ylabel('TPR')
        elif self.ctrl_trajectory_mode == 'Recl-Prec':
            self.trace_plot.set_xlabel('Recall')
            self.trace_plot.set_ylabel('Precision')
        elif self.ctrl_trajectory_mode == 'Spec-Sens':
            self.trace_plot.set_xlabel('Specificity')
            self.trace_plot.set_ylabel('Sensitivity ')
        else:
            raise ValueError

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


#   Used for obtain interpolated color
#   value = 0 -> 1, color = color_1 -> color_2
def color_interpolation(color_1, color_2, values):
    color_1 = np.reshape(color_1, (1, 3))
    color_2 = np.reshape(color_2, (1, 3))
    values = np.reshape(values, (-1, 1))
    return color_1 * (1 - values) + color_2 * values
