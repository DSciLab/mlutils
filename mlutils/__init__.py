import os
import torch
import torch.cuda
import numpy as np
import random

from .trainer import Trainer
from .log import Log
from .dashboard import Dashobard, show_dashboard
from .meter import AverageMeter
from .saver import Saver
from .imread import imread
from .container import DataContainer
from .utils import LogitToPreds, MultiLoss, threhold_seg
from .model import EMAModel
from .inspector import Inspector


def init(opt):
    if hasattr(opt, 'device'):
        if not isinstance(opt.device, list):
            device = [opt.device]
        else:
            device = opt.device

        device = ','.join([str(d) for d in device])
        os.environ['CUDA_VISIBLE_DEVICES'] = device
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if opt.get('test', False):
        saver = Saver(opt)
        saver.load_cfg(opt)

    if hasattr(opt, 'seed'):
        random.seed(opt.seed)
        torch.seed(opt.seed)
        torch.cuda.seed(opt.seed)
        np.random.seed(opt.seed)
