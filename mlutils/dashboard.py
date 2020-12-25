import torch
from collections import defaultdict
from .log import Log
from visdom import Visdom
import numpy as np
import socket
from .log import Log
from .shell import Shell


def get_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('localhost', 0))
    _, port = s.getsockname()
    s.close()
    return port


def get_hostname():
    return socket.gethostname()


def regist_win(func):
    def __fn(*args, **kwargs):
        obj = args[0]
        title = args[1]
        win = func(*args, **kwargs)
        obj._regist_win(title, win)
        return win
    return __fn


class Dashobard(object):
    def __init__(self, opt):
        self.env = opt.id
        self.server_port = opt.get('visdom_port', get_free_port())
        self.opt = opt
        self.hostname = get_hostname()

        self.training = True
        self.win_dict = defaultdict(lambda: None)
        self.server_pid = None
        if opt.get('visdom_server', False):
            self._start_server_in_app = True
            self.start_server()
        else:
            self._start_server_in_app = False
        if opt.get('dashboard', False):
            self.enabled = False
        else:
            self.enabled = True
            self.viz = Visdom(port=self.port)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def start_server(self):
        cmd = f'python -m visdom.server -port {self.port} '
        cmd += f'-hostname 0.0.0.0'
        code, pid, stdout, stderr = Shell.run()
        self.server_pid = pid
        Log.info(f'http://{self.hostname}:{self.port}')

    def kill_server(self):
        if self._start_server_in_app:
            Shell.kill9(self.server_pid)

    def get_title(self, title):
        if self.training:
            prefix = 'train'
        else:
            prefix = 'eval'

        return f'{prefix}/{title}'

    def _regist_win(self, title, win):
        if self.win_dict[title] is None:
            self.win_dict[title] = win

    @regist_win
    def add_image(self, title, image):
        if self.enabled is False:
            Log.warn('try to show image, while dashboard is disabled')
            return
        title = self.get_title(title)
        if image.dim() == 4:
            image = image[0, :, :, :]

        return self.vis.image(image,
                              opts={'title': title},
                              env=self.env,
                              win=self.win_dict[title])

    def add_image_dict(self, image_dict):
        for title, image in image_dict.items():
            self.add_image(title, image)

    @regist_win
    def add_line(self, title, X, Y):
        if self.enabled is False:
            Log.warn('try to plot line, while dashboard is disabled')
            return
        return self.vis.line(X=X, Y=Y,
                             opts={'title': title},
                             env=self.env,
                             win=self.win_dict[title],
                             update='append')

    def add_trace(self, title, epoch, data):
        if isinstance(data, torch.Tensor) or \
            isinstance(data, np.ndarray):
            data = data.mean()
        self.add_line(title, epoch, data)

    def add_trace_dict(self, epoch, data_dict):
        for title, data in data_dict.items():
            self.add_trace(title, epoch, data)
