# !/usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    train_confusion_path = './confusion_example/confusions_train.pkl'
    valid_confusion_path = './confusion_example/confusions_validate.pkl'

    from pinglib.interactive.confusion_visualizer import Confusion_Visualizer
    conf_visual = Confusion_Visualizer(train_confusion_path, valid_confusion_path,
                                       class_names=['Tumor', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX'])


if __name__ == "__main__":
    main()
