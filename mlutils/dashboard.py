import time
import torch
from collections import defaultdict
from .log import Log
from visdom import Visdom
import numpy as np
import socket
from IPython.display import IFrame
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
        title = obj.get_title(title)
        win = func(*args, **kwargs)
        obj._regist_win(title, win)
        return win
    return __fn


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Dashobard(object, metaclass=Singleton):
    MAX_TRY_CONNECT_SERVER = 10
    IMG_FREQ = 11

    def __init__(self, opt):
        self.img_freq = opt.get('img_freq', self.IMG_FREQ)
        self.image_step = 0
        self.image_chan = opt.image_chan
        self.env = opt.id
        self.port = opt.get('dashboard_port', get_free_port())
        self.opt = opt
        self.opt_html = None
        self.hostname = get_hostname()

        self.training = True
        self.win_dict = defaultdict(lambda: None)
        self.server_pid = None
        self.epoch = np.array([0])

        if opt.get('dashboard_server', False):
            self._start_server_in_app = True
            self.start_server()
        else:
            if opt.get('dashboard', False):
                # if dashboard is enabled.
                if opt.get('dashboard_port', None) is None:
                    # dashboard port not set.
                    Log.error(f'visdom server port not set.')
            self._start_server_in_app = False

        if not opt.get('dashboard', False):
            Log.info('Dashboard disabled.')
            self.enabled = False
            self.viz = None
        else:
            Log.info('Dashboard enabled.')
            self.enabled = True
            if self._start_server_in_app:
                try_cnt = 0
                while True:
                    try_cnt += 1
                    try:
                        self.viz = Visdom(port=self.port,
                                          raise_exceptions=True)
                    except ConnectionError as e:
                        if try_cnt > self.MAX_TRY_CONNECT_SERVER:
                            raise e
                        Log.info(f'[{try_cnt}/{self.MAX_TRY_CONNECT_SERVER}] '
                                 'Connect to visdom server error, '
                                 'auto try 5 sec. leter.')
                        time.sleep(5)
                        continue
                    else:
                        break
            else:
                self.viz = Visdom(port=self.port, raise_exceptions=True)
            self.address = f'http://{self.hostname}:{self.port}/env/{self.opt.id}'
            Log.info('initiated dashboard.')
            Log.info(f'Address: {self.address}')
            self.show_status()

    def opt_to_html(self):
        to_show_list = ['lr', 'group', 'dataset', 'arch', 'device']
        if self.opt_html is None:
            output = ''
            for key in to_show_list:
                value = self.opt.get(key, None)
                if value is not None:
                    output += f'<h4>{key}: {value}</h4>'
            self.opt_html = output

        epoch_html = f'<h4>Epoch: {self.epoch[0]}</h4>'
        return self.opt_html + epoch_html

    def show_status(self):
        html = self.opt_to_html()
        win = self.viz.text(html,
                            env=self.env,
                            opts={'title': 'status'},
                            win=self.win_dict['status'])

        if  self.win_dict['status'] is None:
            self.win_dict['status'] = win

    def step(self):
        self.epoch += 1
        self.show_status()

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def start_server(self):
        Log.info('starting dashboard server...')
        cmd = f'python -m visdom.server -port {self.port} '
        cmd += f'--hostname 0.0.0.0'
        code, pid, stdout, stderr = Shell.run(cmd)
        Log.debug(f'dashboard server code: {code}')
        Log.debug(f'dashboard server stdout: {stdout}')
        Log.debug(f'dashboard server stderr: {stderr}')
        self.server_pid = pid
        Log.info(f'Dashboard: http://{self.hostname}:{self.port}/env/{self.opt.id}')

    def kill_server(self):
        if self._start_server_in_app:
            Shell.kill9(self.server_pid)

    def get_title(self, title):
        if self.training:
            prefix = 'train'
        else:
            prefix = 'eval'

        return f'{prefix}_{title}'

    def _regist_win(self, title, win):
        if self.win_dict[title] is None:
            self.win_dict[title] = win

    @regist_win
    def add_image(self, title, image, rgb=False):
        self.image_step += 1
        if self.image_step % self.img_freq != 0:
            return

        if self.enabled is False:
            Log.warn('Try to show image, while dashboard is disabled')
            return
        title = self.get_title(title)
        if self.image_chan == 3 or rgb:
            if image.ndim == 4:
                image = image[0, :, :, :]
        else:
            # image chan == 1
            if image.ndim == 4:
                image = image[0, :, :, :]
            else:
                # dim == 3
                image = image[0, :, :]
                if isinstance(image, torch.Tensor):
                    image = torch.unsqueeze(image, 0)
                else:
                    # ndarray
                    image = np.expand_dims(image, 0)

        if isinstance(image, torch.Tensor) and \
            image.dtype is torch.int:
            image = image.type(torch.float)
        elif isinstance(image, np.ndarray) and \
            'int' in str(image.dtype):
            image = image.astype(np.float)

        return self.viz.image(image,
                              opts={'title': title},
                              env=self.env,
                              win=self.win_dict[title])

    def add_image_dict(self, image_dict):
        for title, image in image_dict.items():
            self.add_image(title, image)

    @regist_win
    def add_line(self, title, X, Y, step=None):
        if self.enabled is False:
            Log.warn('Try to plot line, while dashboard is disabled')
            return

        if isinstance(Y, (int, float, np.float, np.int)):
            Y = np.array([Y])

        if isinstance(Y, torch.Tensor):
            if Y.ndim == 0:
                Y = torch.unsqueeze(Y, 0)
        elif not isinstance(Y, np.ndarray):
            Log.error(f'Unrecognized meter value {Y}/{type(Y)}.')

        title = self.get_title(title)
        if self.win_dict[title] is None:
            return self.viz.line(X=X, Y=Y,
                                opts={'title': title},
                                env=self.env,
                                win=self.win_dict[title])
        else:
            return self.viz.line(X=X, Y=Y,
                                opts={'title': title},
                                env=self.env,
                                win=self.win_dict[title],
                                update='append')

    def add_trace(self, title, data, step=None):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
        if step is None:
            step = self.epoch
        else:
            step = np.array([step])
        self.add_line(title, step, data)

    def add_trace_dict(self, data_dict, step=None):
        for title, data in data_dict.items():
            self.add_trace(title, data, step)

    def add_meter(self, meter, step=None):
        value = meter.latest
        self.add_trace(meter.name, value, step)


def show_dashboard(opt, width=1300, height=1300):
    dashboard = Dashobard(opt)
    width = opt.get('dashboard_width', width)
    height = opt.get('dashboard_height', height)
    return IFrame(f'{dashboard.address}',
                  width=width, height=height)
