from keras import backend as K
from .losses import NB



class OutlierLoss():
    def __init__(self):
        pass

    def __call__(self, y_true, pred_mean):
        counts = y_true[0].flatten()
        idx = y_true[1].flatten()
        pred_mean = pred_mean.flatten()
        nb = NB(out_idx=idx)
        loss_res = K.eval(nb.loss(counts,pred_mean))
        return loss_res


class CorrectedCorr():
    def __init__(self):
        pass

    def __call__(self, y_true, pred_mean):
        counts = y_true[0]
        ev = Evaluation(counts, pred_mean)
        metric = ev.mean_corrected_corr
        return metric
