# !/usr/bin/env python
# -*- coding: utf-8 -*-

def main():
    train_confusion_path = './confusion_example/confusions_train.pkl'
    valid_confusion_path = './confusion_example/confusions_validate.pkl'

    from pinglib.interactive.confusion_visualization import Confusion_Visualization
    conf_visual = Confusion_Visualization(train_confusion_path, valid_confusion_path,
                                          class_names=['Tumor', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX', 'XX'])


if __name__ == "__main__":
    main()
