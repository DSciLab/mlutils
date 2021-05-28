import time
import threading as T
import queue
from cfg import Opts
from typing import Any, List, TypeVar, Callable
from mlutils.log import Log

Queue = TypeVar('Queue')
S_CallBack = Callable[[TypeVar('ParallelTrainer')], None]
GR_CallBack = Callable[[TypeVar('ParallelTrainer'), List[Any]], None]

class Msg(object):
    def __init__(self) -> None:
        super().__init__()
        self.msg_type = None
        self.data


class Worker(object):
    def __init__(self) -> None:
        super().__init__()
        self.id = None
        self.msg_queue = None
        self.device_id = None

    def set_worker_id(self, worker_id):
        self.id = worker_id

    def set_msg_queue(self, msg_queue: Queue) -> None:
        self.msg_queue = msg_queue

    def set_device_id(self, device_id: int) -> None:
        self.device_id = device_id

    def _work(self) -> None:
        # before work
        result = self.work()
        self._set_result(result)
        # after work

    def _set_result(self, result: Any) -> None:
        self.msg_queue.put(result)

    def report(self) ->str:
        raise NotImplementedError

    def init(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def work(self):
        raise NotImplementedError


class ParallelTrainer(object):
    def __init__(self, opt: Opts) -> None:
        super().__init__()
        self.opt = opt
        self.msg_queue = queue.Queue()
        self.workers = []
        self.running_workers = []
        self.done_workers = []
        self.nprocs = None
        self.divece_list = opt.device
        self.worker_lock = T.Lock()
        self.device_lock = T.Lock()
        self.device_query_lock = T.Lock()
        self.thread_list = []
        self.start_callback_list = []
        self.get_result_callback_list = []
        self.device_ids = list(range(len(self.divece_list)))

    def append(self, worker: Worker) -> None:
        worker.set_msg_queue(self.msg_queue)
        self.workers.append(worker)

    def get_device(self, wait: bool=True) -> int:
        while True:
            self.device_query_lock.acquire()
            if len(self.device_ids) == 0:
                if not wait:
                    self.device_query_lock.release()
                    raise RuntimeError(
                        f'Resource [device] has been runs out.')
                time.sleep(1) # wait 1 sec.
                self.device_query_lock.release()
                continue
            else:
                break

        self.device_lock.acquire()
        device_id = self.device_ids.pop()
        self.device_lock.release()
        self.device_query_lock.release()
        return device_id

    def return_device(self, device_id: int) -> None:
        self.device_lock.acquire()
        self.device_ids.append(device_id)
        self.device_lock.release()

    def get_worker(self) -> Worker:
        self.worker_lock.acquire()
        if len(self.workers) == 0:
            raise RuntimeError(
                f'Worker has been runs out, '
                f'len(workers)={len(self.workers)}')
        worker = self.workers.pop()
        self.worker_lock.release()
        return worker

    def set_running_workers(self, worker: Worker) -> None:
        self.worker_lock.acquire()
        self.running_workers.append(worker)
        self.worker_lock.release()

    def set_done_workers(self, worker: Worker) -> None:
        self.worker_lock.acquire()
        self.running_workers.remove(worker)
        self.done_workers.append(worker)
        self.worker_lock.release()

    def wake_worker(self, index: int) -> None:
        device_id = self.get_device(wait=True)
        worker = self.get_worker()
        worker.set_device_id(device_id)
        worker.set_worker_id(index)
        self.set_running_workers(worker)
        worker._work()
        self.return_device(worker.device_id)
        self.set_done_workers(worker)

    def collect_report(self) -> str:
        string = ''
        for worker in self.running_workers:
            try:
                string += worker.report()
            except NotImplementedError:
                pass
            string += ' | '
        return string

    def start(self) -> None:
        for callback in self.start_callback_list:
            callback(self)

        self.nprocs = len(self.workers)
        for i in range(self.nprocs):
            t = T.Thread(target=self.wake_worker, args=(i,))
            t.start()
            self.thread_list.append(t)
    
    def progress_bar(self) -> None:
        string = f'[Remaining: {len(self.workers)}]' +\
                 f'[Running: {len(self.running_workers)}]' +\
                 f'[Done: {len(self.done_workers)}] | '
        string += self.collect_report()
        Log.progress(string)

    def wait_result(self) -> List[Any]:
        results = []
        cnt = 0
        while True:
            self.progress_bar()
            try:
                result = self.msg_queue.get(timeout=1)
            except queue.Empty:
                continue
            results.append(result)
            for callback in self.get_result_callback_list:
                callback(self, results)
            cnt += 1
            if cnt >= self.nprocs:
                break

        for t in self.thread_list:
            t.join()
        return results

    def register_get_result_hook(self, fn: S_CallBack) -> None:
        self.get_result_callback_list.append(fn)

    def register_start_hook(self, fn: GR_CallBack) -> None:
        self.start_callback_list.append(fn)


if __name__ == '__main__':
    class W(Worker):
        def work(self):
            import numpy as np
            time.sleep(int(np.random.randint(1, 5, 1)))
            print('work', self.id)
            # assert False
            return 1

    class Opts:
        def __init__(self) -> None:
            self.device = [1,2,3]

    opt = Opts()
    pt = ParallelTrainer(opt)
    for i in range(5):
        pt.append(W())
    
    pt.start()
    print(pt.wait_result())
