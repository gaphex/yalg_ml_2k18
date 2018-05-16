import os
import pandas as pd

from datetime import datetime
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

class LossHistory(Callback):
    def __init__(self, fpath=None):
        self.fpath = fpath
        
    def on_train_begin(self, logs={}):
        self.losses = []
        self.alllosses = []

    def on_batch_end(self, batch, logs={}):
        logs['timestamp'] = get_timestamp()
        self.losses.append(logs)
        
    def on_epoch_end(self, batch, logs={}):
        logs['timestamp'] = get_timestamp()
        self.losses.append(logs)
        columns=['acc', 'loss', 
                 'val_acc', 'val_loss', 
                 'batch', 'size', 'timestamp']
        if self.fpath:
            if os.path.exists(self.fpath):
                outloss = pd.read_csv(self.fpath)[columns]
            else:
                outloss = pd.DataFrame(columns=columns)
            for i, loss in enumerate(self.losses):
                outloss = outloss.append(loss, ignore_index=True)
                    
            outloss.to_csv(self.fpath)
        self.alllosses += self.losses
        self.losses = []
    
    
class AUC_Saver(Callback):
    def __init__(self, fpath, valdata, batchsize=512, min_auc=0.0, save_best_only=True):
        self.sbo = save_best_only
        self.savepath = fpath
        self.min_auc = min_auc
        self.batchsize = batchsize
        self.valdata = valdata
    
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.valdata[0],batch_size=self.batchsize)
        cur_auc = roc_auc_score(self.valdata[1], y_pred)
        self.aucs.append(cur_auc)
        
        if cur_auc >= max(self.aucs):
            print("\nNew best model with AUC = {}, saving to {}".format(cur_auc, self.savepath))
            self.model.save_weights(self.savepath)
        if cur_auc>=self.min_auc and not self.sbo:
            self.model.save_weights(self.savepath+'_AUC='+str(cur_auc)[:7])
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return
    
    
class LogLoss_Saver(Callback):
    def __init__(self, fpath, valdata, batchsize=512):
        self.savepath = fpath
        self.batchsize = batchsize
        self.valdata = valdata
    
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):

        y_pred = self.model.predict(self.valdata[0],batch_size=self.batchsize)
        cur_loss = log_loss(self.valdata[1], y_pred)
        self.aucs.append(cur_loss)
        
        if cur_loss <= min(self.aucs):
            print("\nNew best model with LogLoss = {}, saving to {}".format(cur_loss, self.savepath))
            self.model.save_weights(self.savepath)

        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return


def get_timestamp():
    return datetime.strftime(datetime.now(), "%j %T")

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
        
def maybe_mkdir(dirpath):
    try:
        os.mkdir(dirpath)
    except Exception as e:
        pass

