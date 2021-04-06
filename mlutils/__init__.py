import os
import torch
import torch.cuda
import numpy as np
import random

from .trainer import Trainer, detach_cpu
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

        # check gpu
        if hasattr(opt, 'gpu_black_list'):
            assert  hasattr(opt, 'hostname'), f'hostname not found.'
            gpu_black_list = opt.gpu_black_list[opt.hostname]

            for gpu_id in device:
                if gpu_id in gpu_black_list:
                    raise RuntimeError(
                        f'GPU_{gpu_id} in GPU black list of {opt.hostname}.')

        device = ','.join([str(d) for d in device])
        os.environ['CUDA_VISIBLE_DEVICES'] = device

    if opt.get('test', False):
        saver = Saver(opt)
        saver.load_cfg(opt)

    if hasattr(opt, 'seed'):
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        np.random.seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
