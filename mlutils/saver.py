import os
import shutil
from typing import Any
import torch
from .log import Log
from .container import DataContainer
from .meter import KFoldMeter
import pickle


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class Saver(object):
    DEFAULT_ROOT = '.saver'
    METER_LOG = 'meters_{0}.log'
    LATEST_STATE = 'latest_{0}.pth'
    BEST_STATE = 'best_{0}.pth'
    CFG_FILE = 'cfg.yaml'
    CONTAINER_FILE = '{1}_container_latest_{0}.pickle'
    KFOLD_METER = 'k_fold_meter.pickle'

    def __init__(self, opt):
        self.test = opt.get('testing', False)
        self.saver_dir = os.path.join(self.DEFAULT_ROOT, opt.id)
        self.cfg_path = os.path.join(self.saver_dir, self.CFG_FILE)
        self.curr_fold = 0
        if self.test is False:
            self.create_saver_dir(opt, self.saver_dir, self.DEFAULT_ROOT)
        # Log.info('initiated saver.')

    @property
    def latest_path(self):
        latest_name = self.get_curr_fold_file(self.LATEST_STATE)
        return os.path.join(self.saver_dir, latest_name)

    @property
    def best_path(self):
        best_state = self.get_curr_fold_file(self.BEST_STATE)
        return os.path.join(self.saver_dir, best_state)

    @property
    def meters_path(self):
        meter_log = self.get_curr_fold_file(self.METER_LOG)
        return os.path.join(self.saver_dir, meter_log)

    def set_fold(self, k):
        self.curr_fold = k

    def get_curr_fold_file(self, name, *args, k=None):
        if k is None:
            k = self.curr_fold
        return name.format(k, *args)

    def create_saver_dir(self, opt, path, root):
        if not os.path.exists(root):
            try:
                os.mkdir(root)
            except FileExistsError as e:
                pass

        if os.path.exists(path):
            if opt.get('train_mod', 'split') == 'k_fold':
                return
            if not opt.get('override', False) and not opt.get('dist', False):
                raise RuntimeError(
                    f'saver path ({path}) exists, '
                    'set overrde=True to override')
            else:
                shutil.rmtree(path)

        try:
            os.mkdir(path)
        except FileExistsError as e:
            pass

    def save_object(self, obj: Any, name: str) -> None:
        path = os.path.join(self.saver_dir, name)
        save_pickle(obj, path)

    def load_object(self, name: str) -> Any:
        path = os.path.join(self.saver_dir, name)
        return load_pickle(path)

    def save_container(self, container, best=False):
        assert isinstance(container, DataContainer)
        container_name = self.get_curr_fold_file(self.CONTAINER_FILE,
                                                 container.name)
        container_path = os.path.join(self.saver_dir, container_name)
        container.dump(container_path)
        if best:
            # Log.info(f'Save best container [{container.name}].')
            best_path = container_path.replace('latest', 'best')
            shutil.copyfile(container_path, best_path)

    def load_container(self, name, best=False, k=None):
        container_name = self.get_curr_fold_file(self.CONTAINER_FILE, name, k=k)
        container_path = os.path.join(self.saver_dir, container_name)
        container = DataContainer(name)
        if best:
            container_path = container_path.replace('latest', 'best')
        container.load(container_path)
        return container

    def save_k_fold_meter(self, meter):
        assert isinstance(meter, KFoldMeter)
        k_fold_meter_path = os.path.join(self.saver_dir, self.KFOLD_METER)
        meter.dump(k_fold_meter_path)

    def load_k_fold_meter(self):
        k_fold_meter_path = os.path.join(self.saver_dir, self.KFOLD_METER)
        meter = KFoldMeter()
        meter.load(k_fold_meter_path)
        return meter

    def save_state_dict(self, state_dict, best=False):
        if self.test is True:
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
        if self.test is True:
            # Do nothing on test stage
            return

        # Log.info('Save best model.')
        shutil.copyfile(self.latest_path, self.best_path)

    def _save_latest_state_dict(self, state_dict):
        if self.test is True:
            # Do nothing on test stage
            return

        if os.path.exists(self.latest_path):
            os.remove(self.latest_path)
        torch.save(state_dict, self.latest_path)

    def _load_best_state_dict(self):
        if os.path.exists(self.best_path):
            Log.info(f'Loading state_dict from {self.best_path}')
            state = torch.load(self.best_path)
            return state
        else:
            raise RuntimeError(f'State not found [{self.best_path}].')

    def _load_latest_state_dict(self):
        if os.path.exists(self.latest_path):
            Log.info(f'Loading state_dict from {self.latest_path}')
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
