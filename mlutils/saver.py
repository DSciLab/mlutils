import os
import shutil
import torch
from .log import Log


class Saver(object):
    DEFAULT_ROOT = '.saver'
    METER_LOG = 'meters.log'
    LATEST_STATE = 'latest.pth'
    BEST_STATE = 'best.pth'
    CFG_FILE = 'cfg.yaml'

    def __init__(self, opt):
        self.test = opt.get('test', False)
        saver_root = opt.get('saver_root', self.DEFAULT_ROOT)
        saver_dir = os.path.join(saver_root, opt.id)
        self.latest_path = os.path.join(saver_dir, self.LATEST_STATE)
        self.best_path = os.path.join(saver_dir, self.BEST_STATE)
        self.meters_path = os.path.join(saver_dir, self.METER_LOG)
        self.cfg_path = os.path.join(saver_dir, self.CFG_FILE)
        if not self.test:
            self.create_saver_dir(opt, saver_dir, saver_root)
        Log.info('initiated saver.')

    def create_saver_dir(self, opt, path, root):
        if not os.path.exists(root):
            os.mkdir(root)

        if os.path.exists(path):
            if not opt.get('override', False):
                raise RuntimeError(
                    f'saver path ({path}) exists, '
                    'set overrde=True to override')
            else:
                shutil.rmtree(path)
        os.mkdir(path)

    def save_state_dict(self, state_dict, best=False):
        if self.test:
            # Do nothing on test stage
            return

        self._save_latest_state_dict(state_dict)
        if best:
            self._save_best_state_dict()

    def load_state_dict(self, model):
        if model == 'best':
            return self._load_best_state_dict()
        elif model == 'latest':
            return self._load_latest_state_dict()
        else:
            raise RuntimeError(f'Unrecognized model ({model}).')

    def _save_best_state_dict(self):
        if self.test:
            # Do nothing on test stage
            return

        Log.info('Save best model.')
        shutil.copyfile(self.latest_path, self.best_path)

    def _save_latest_state_dict(self, state_dict):
        if self.test:
            # Do nothing on test stage
            return

        if os.path.exists(self.latest_path):
            os.remove(self.latest_path)
        torch.save(state_dict, self.latest_path)

    def _load_best_state_dict(self):
        if os.path.exists(self.best_path):
            state = torch.load(self.best_path)
            return state
        else:
            raise RuntimeError(f'State not found [{self.best_path}].')

    def _load_latest_state_dict(self):
        if os.path.exists(self.latest_path):
            state = torch.load(self.latest_path)
            return state
        else:
            raise RuntimeError(f'State not found [{self.latest_path}].')

    def save_meters(self, epoch, *meters):
        if self.test:
            # Do nothing on test stage
            return

        with open(self.meters_path, 'a') as f:
            for meter in meters:
                f.write(str(meter) + '\n')
            f.write(f'======== {epoch} ==========\n')

    def save_cfg(self, opt):
        if self.test:
            # Do nothing on test stage
            return

        opt.dump(self.cfg_path)

    def load_cfg(self, opt):
        opt.load(self.cfg_path)
