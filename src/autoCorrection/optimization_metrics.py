from keras import backend as K
import tensorflow as tf
from .losses import NB



class OutlierLoss():
    def __init__(self):
        pass

    def __call__(self, y_true, pred_mean):
        counts = y_true[0].flatten()
        idx = y_true[1].flatten()
        pred_mean = pred_mean.flatten()
        nb = NB(out_idx=idx)
        nb.theta = tf.Variable([np.float32(25.0)], dtype=tf.float32, name='theta')
        sess = tf.Session()
        K.set_session(sess)
        sess.run(tf.global_variables_initializer())
        with sess.as_default():
            loss_res=nb.loss(counts,pred_mean+1e-10).eval()
        return loss_res


class CorrectedCorr():
    def __init__(self):
        pass

    def __call__(self, y_true, pred_mean):
        counts = y_true[0]
        ev = Evaluation(counts, pred_mean)
        metric = ev.mean_corrected_corr
        return metric
