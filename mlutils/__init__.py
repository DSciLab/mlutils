import os
import random
import datetime

from .trainer import Trainer, detach_cpu
from .log import Log
from .dashboard import Dashobard, show_dashboard
from .meter import AverageMeter
from .saver import Saver
from .imread import imread
from .container import DataContainer
from .utils import LogitToPreds, MultiLoss,\
                   threhold_seg, ProgressSign,\
                   load_pickle, deep_supervised_weight
from .model import EMAModel
from .inspector import Inspector
from .meter import KFoldMeter



def init(opt):
    import torch
    import torch.cuda
    import numpy as np

    if hasattr(opt, 'device'):
        if not isinstance(opt.device, list):
            device = [opt.device]
        else:
            device = opt.device

        # check gpu
        if device[0] != -1:
            if hasattr(opt, 'gpu_black_list'):
                assert  hasattr(opt, 'hostname'), f'hostname not found.'
                gpu_black_list = opt.gpu_black_list[opt.hostname]

                for gpu_id in device:
                    if gpu_id in gpu_black_list:
                        raise RuntimeError(
                            f'GPU_{gpu_id} in GPU black list of {opt.hostname}.')

            device = ','.join([str(d) for d in device])
            os.environ['CUDA_VISIBLE_DEVICES'] = device

    if opt.get('testing', False) is True:
        saver = Saver(opt)
        saver.load_cfg(opt)

    if hasattr(opt, 'seed'):
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
