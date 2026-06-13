
import numpy as np

from matplotlib import pyplot

def plot_traininghistory(history, loss_type = "Categorical_"):

    # plot loss during training
    pyplot.subplot(211)
    pyplot.title('Loss / {0} Cross Entropy'.format(loss_type))
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['{0}accuracy'.format(loss_type.lower())], label='train')
    pyplot.plot(history.history['val_{0}accuracy'.format(loss_type.lower())], label='test')
    pyplot.legend()
    pyplot.show()