import threading
import psutil



class Spy(object):
    def __init__(self, pid_names, interval=0.5):
        self._pid_names = pid_names
        self._interval = interval
        self._processes = {}
        self._threads = []
        self._lock = threading.Lock()
        self._running = False

    def start(self):
        if self._running:
            raise RuntimeError('Spy is running, can not start')
        self._running = True
        for pid, name in self._pid_names.items():
            t = threading.Thread(target=self._monitor, args=(self, pid, name))
            t.start()
            self._threads.append(t)

    def stop(self):
        self._running = False
        for t in self._threads:
            t.join()
        self._threads = []

    def scalars(self, attrs):
        values = {}
        with self._lock:
            for k in attrs:
                d = values[k] = {}
                for pid, status in self._processes.items():
                    id = f'{status["name"]}:{pid}'
                    d[id] = status[k]
        return values

    @staticmethod
    def _monitor(self, pid, name):
        L = self._lock
        processes = self._processes
        p = psutil.Process(pid)
        while self._running:
            status = p.as_dict()
            status['name'] = name
            status['cpu_percent'] = p.cpu_percent(self._interval)
            with L:
                processes[pid] = status