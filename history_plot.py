#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def set_val_train(hist_dict):
    on_off1 = hist_dict.get('val_loss')
    on_off2 = hist_dict.get('accuracy')
    if on_off1:
        if on_off2:
            train_loss = hist_dict['loss']
            train_acc  = hist_dict['accuracy']
            val_loss = hist_dict['val_loss']
            val_acc  = hist_dict['val_accuracy']
        else:
            train_loss = hist_dict['loss']
            train_acc  = hist_dict['acc']
            val_loss = hist_dict['val_loss']
            val_acc  = hist_dict['val_acc']
    else:
        if on_off2:
    	    train_loss = hist_dict['loss']
    	    train_acc  = hist_dict['accuracy']
        else:
    	    train_loss = hist_dict['loss']
    	    train_acc  = hist_dict['acc']
        shapes = len(train_acc)
        nan_array = np.empty((shapes,))
        nan_array.fill(np.nan)
        val_loss = nan_array
        val_acc = nan_array 
    return train_loss, train_acc, val_loss, val_acc

def history_plot(history, png_file_name='loss_acc_plot.png'):
	hist_dict = history.history
	train_loss,train_acc,val_loss,val_acc = set_val_train(hist_dict)
	x = range(1, len(train_acc)+1) 
	fig, loss_ax = plt.subplots(figsize=(8,6))
	acc_ax = loss_ax.twinx()
	loss_ax.plot(x,train_loss, 'y', label='train loss')
	loss_ax.plot(x,val_loss, 'r', label='validation loss')
	acc_ax.plot(x, train_acc, 'b', label='train acc')
	acc_ax.plot(x, val_acc, 'g', label='val acc')
	for label in acc_ax.get_yticklabels(): # y축 tick 색깔 지정
		label.set_color("blue")
	loss_ax.set_xlabel('epoch',size=20)
	loss_ax.set_ylabel('loss',size=20)
	acc_ax.set_ylabel('accuray', color='blue',size=20)
	loss_ax.legend(loc='upper right')
	acc_ax.legend(loc='lower right')
	fig.savefig(png_file_name)
	plt.show()
