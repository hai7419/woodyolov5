import logging
import logging.config
from pathlib import Path
import os
import numpy as np
import torch
from copy import deepcopy
import math

import random
import torch

def set_logging(name='my-logger'):
    # sets up logging for the given name
    level = logging.INFO
    # logging.basicConfig(level=logging.DEBUG)
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging()
LOGGER = logging.getLogger('my-logger')




def increment_path(path,exist_ok=False,sep='',mkdir=False):

    """
    output increment path and creak the folder
    input:path      character
          exist_ok
          mkdir     
    output" path    increment path number 2-9999
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        # path, suffix = (path.with_suffix(''),path.suffix) if path.is_file() else (path, '')
        for n in range(2,9999):
            p = f'{path}{sep}{n}'
            if not os.path.exists(p):
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True,exist_ok=True)
    return path


def check_dataset(path):
    """
    output  
    """
    
    return{
        'path':path.parents[0],
        #'train':Path('/content/sample_data/train/'),          #os.path.join(path.parents[0],'/train'),
        #'val':Path('/content/sample_data/val/'),             #             os.path.join(path.parents[0],'/val'),
        'train':os.path.join(path.parents[0],'datasets/wood/images/train'),
        'val':os.path.join(path.parents[0],'datasets/wood/images/val'),
        
        # 'train':os.path.join(path.parents[0],'datasets\\pcb\\images\\train'),
        # 'val':os.path.join(path.parents[0],'datasets\\pcb\\images\\val'),
        'test':None,
        'names':{0:'Quartzity',1:'Live_Knot',2:'Marrow',3:'resin',4:'Dead_Knot',5:'knot_with_crack',6:'Knot_missing',7:'Crack'},
        'nc':8
    }
    

def labels_to_class_weights(labels, nc=80):
    """
    Get class weights (inverse frequency) from training labels
    input   labels [] list [][]   classs xywh
            nc  number class
    output  class weights    number more  weights less
    
    """
    
    
    if labels[0] is None:  # no labels loaded
        return torch.Tensor()

    labels = np.concatenate(labels, 0)  
    classes = labels[:, 0].astype(int)  # labels = [class xywh]
    weights = np.bincount(classes, minlength=nc)  # occurrences per class

    weights[weights == 0] = 1  # replace empty bins with 1
    weights = 1 / weights  # number of targets per class     number more  weights less
    weights /= weights.sum()  # normalize
    return torch.from_numpy(weights).float()



class EarlyStopping:
    # YOLOv5 simple early stopper
    def __init__(self, patience=30):
        self.best_fitness = 0.0  # i.e. mAP
        self.best_epoch = 0
        self.patience = patience  # epochs to wait after fitness stops improving to stop
        self.possible_stop = False  # possible stop may occur next epoch

    def __call__(self, epoch, fitness):
        if fitness >= self.best_fitness:  # >= 0 to allow for early zero-fitness stage of training
            self.best_epoch = epoch
            self.best_fitness = fitness
        delta = epoch - self.best_epoch  # epochs without improvement
        self.possible_stop = delta >= (self.patience - 1)  # possible stop may occur next epoch
        stop = delta >= self.patience  # stop training if patience exceeded
        if stop:
            LOGGER.info(f'Stopping training early as no improvement observed in last {self.patience} epochs. '
                        f'Best results observed at epoch {self.best_epoch}, best model saved as best.pt.\n'
                        f'To update EarlyStopping(patience={self.patience}) pass a new patience value, '
                        f'i.e. `python train.py --patience 300` or use `--patience 0` to disable EarlyStopping.')
        return stop
    


def fitness(x):
    """
        fitness index 
    
    """
    w = [0.0, 0.0, 0.1, 0.9] #weight for [p,r,mAP@0.5,mAP@.5:0.95]
    return (x[:,:4]*w).sum(1)


class ModelEMA:
    """

    
    """
    def __init__(self, model,decay=0.9999,tau=2000,updates=0) -> None:
        self.ema = deepcopy(model).eval()
        self.updates = updates
        self.decay = lambda x: decay*(1-math.exp(-x / tau)) 
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        self.updates += 1
        d = self.decay(self.updates)

        msd = model.state_dict()
        for k,v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v *= d
                v += (1 -d)*msd[k].detach()
    
    def update_attr(self,model,include=(),exclude=('process_group','reducer')):
        for k,v in model.__dict__.items():
            if(len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema,k,v)



def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)
    torch.cuda.manual_seed(seed=seed)
    torch.cuda.manual_seed_all(seed=seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)





