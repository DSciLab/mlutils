import subprocess
import shlex
from .log import Log


class Shell(object):
    PROCESSES = {}
    @classmethod
    def run(cls, cmd):
        cmd_args = shlex.split(cmd)
        try:
            proc = subprocess.Popen(cmd_args,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        except Exception as e:
            Log.info(f'Cann\'t run [{cmd}]')
            raise e

        pid = proc.pid
        cls.PROCESSES[pid] = proc
        stdout = b''
        stderr = b''
        try:
            stdout, stderr = proc.communicate(timeout=1)
        except subprocess.TimeoutExpired as e:
            cls.PROCESSES.pop(pid)

        code = proc.returncode
        if code is None:
            return 0, pid, stdout, stderr
        else:
            return code, pid, stdout, stderr

    @classmethod
    def kill_all(cls):
        for proc in cls.PROCESSES.values():
            Log.info(f'Kill subprocess {proc.pid}')
            proc.kill()
        cls.PROCESSES = {}

    @classmethod
    def kill(cls, pid):
        proc = cls.PROCESSES.get(pid)
        if proc is not None:
            cls.PROCESSES.pop(pid)
            proc.kill()
        else:
            Log.warn(f'No such pid [{pid}].')

    @classmethod
    def kill9(cls, pid):
        if pid in cls.PROCESSES.keys():
            cls.PROCESSES.pop(pid)
            cmd = f'kill -9 {pid}'
            _, pid, _, _ = cls.run(cmd)
        else:
            Log.warn(f'No such pid [{pid}].')

    @classmethod
    def wait(cls, pid):
        proc = cls.PROCESSES.get(pid)
        if proc is not None:
            cls.PROCESSES.pop(pid)
            proc.wait()
        else:
            Log.warn(f'No such pid [{pid}].')

    @classmethod
    def wait_all(cls):
        while len(cls.PROCESSES) > 0:
            keys = cls.PROCESSES.keys()
            for pid in keys:
                proc = cls.PROCESSES[pid]
                try:
                    proc.wait(1)
                    cls.PROCESSES.pop(pid)
                except subprocess.TimeoutExpired as e:
                    continue
        cls.PROCESSES = {}
