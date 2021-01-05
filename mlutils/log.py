class Log(object):
    LEVEL_LIMIT = 1
    DEBUG = 0
    INFO = 1
    WARN = 2
    ERROR = 3
    FATAL = 4

    @classmethod
    def set_level(cls, level):
        if level in [cls.DEBUG, cls.INFO,
                     cls.WARN, cls.ERROR]:
            cls.LEVEL_LIMIT = level
        else:
            raise ValueError(f'Unrecognized log level [{level}].')

    @classmethod
    def debug(cls, *msg):
        header = '[Debug]: '
        if cls.DEBUG >= cls.LEVEL_LIMIT:
            cls._log(header, *msg)

    @classmethod
    def info(cls, *msg):
        header = '[Info]: '
        if cls.INFO >= cls.LEVEL_LIMIT:
            cls._log(header, *msg)
    
    @classmethod
    def warn(cls, *msg):
        header = '[Warn]: '
        if cls.WARN >= cls.LEVEL_LIMIT:
            cls._log(header, *msg)

    @classmethod
    def error(cls, *msg):
        header = '[Error]: '
        if cls.ERROR >= cls.LEVEL_LIMIT:
            cls._log(header, *msg)

    @classmethod
    def fatal(cls, *msg):
        header = '[Fatal]: '
        if cls.FATAL >= cls.LEVEL_LIMIT:
            cls._log(header, *msg)
        exit(1)

    @classmethod
    def _log(self, header, *msg):
        print(header, *msg)
