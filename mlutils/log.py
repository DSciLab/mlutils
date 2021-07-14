import threading as T


STATUS_PROGRESS = 0
STATUS_NORMAL = 1


class Log(object):
    print_lock = T.Lock()
    latest_status = STATUS_NORMAL

    LEVEL_LIMIT = 1
    DEBUG = 0
    PROGRESS = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    FATAL = 5

    @classmethod
    def set_level(cls, level):
        if level in [cls.DEBUG, cls.INFO,
                     cls.WARN, cls.ERROR]:
            cls.LEVEL_LIMIT = level
        else:
            raise ValueError(f'Unrecognized log level [{level}].')

    @classmethod
    def debug(cls, *msg, **kwargs):
        header = '[Debug]: '
        if cls.DEBUG >= cls.LEVEL_LIMIT:
            cls._log(header, *msg, **kwargs)

    @classmethod
    def info(cls, *msg, **kwargs):
        header = '[Info]: '
        if cls.INFO >= cls.LEVEL_LIMIT:
            cls._log(header, *msg, **kwargs)

    @classmethod
    def progress(cls, msg: str, **kwargs):
        header = '[Progress]: '
        if cls.PROGRESS >= cls.LEVEL_LIMIT:
            cls.print_lock.acquire()
            cls.latest_status = STATUS_PROGRESS
            print(header, msg, end='\r')
            cls.print_lock.release()

    @classmethod
    def warn(cls, *msg, **kwargs):
        header = '[Warn]: '
        if cls.WARN >= cls.LEVEL_LIMIT:
            cls._log(header, *msg, **kwargs)

    @classmethod
    def error(cls, *msg, **kwargs):
        header = '[Error]: '
        if cls.ERROR >= cls.LEVEL_LIMIT:
            cls._log(header, *msg, **kwargs)

    @classmethod
    def fatal(cls, *msg, **kwargs):
        header = '[Fatal]: '
        if cls.FATAL >= cls.LEVEL_LIMIT:
            cls._log(header, *msg, **kwargs)
        exit(1)

    @classmethod
    def _log(cls, header, *msg, **kwargs):
        cls.print_lock.acquire()
        if cls.latest_status == STATUS_PROGRESS:
            print('')
        cls.latest_status = STATUS_NORMAL
        fmt_str = ' '.join([header, *msg])
        print(fmt_str)
        cls.print_lock.release()
