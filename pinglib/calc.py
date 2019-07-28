#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def mean_filter_padded(arr, filter_width=1):
    from scipy.signal import convolve
    #   2的话，在后面加一个，3的话，再在前面加一个……
    add_on_tail = int(filter_width / 2)
    add_on_head = filter_width - 1 - add_on_tail
    arr_padded = np.concatenate([arr[0] * np.ones(add_on_head), arr, arr[-1] * np.ones(add_on_tail)])
    kernel = np.ones(filter_width) / filter_width
    return convolve(arr_padded, kernel, mode='valid')
