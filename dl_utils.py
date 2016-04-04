__author__ = 'Mike'
import threading
import Pyro4
import random

class RepeatedTimer(threading.Thread):
    def __init__(self, interval, func, *args, **kwargs):
        super(RepeatedTimer, self).__init__()
        self.interval = interval
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._stop = threading.Event()
        self.paused = False

    def pause(self):
        self.paused = True
    def resume(self):
        self.paused = False

    def run(self):
        while not self._stop.wait(self.interval):
            if not self.paused:
                self.func(*self.args, **self.kwargs)

    def stop(self):
        self._stop.set()

class MasterWrapper:
    def __init__(self, uri, name, logger):
        self.proxy = None
        self.name = name
        self.logger = logger
        try:
            self.proxy = Pyro4.core.Proxy(uri)
        except Pyro4.errors.CommunicationError, Pyro4.errors.ConnectionClosedError:
            self.proxy._pyroReconnect()
        logger.info("Connected to master")

    def __getattr__(self, item):
        while True:
            try:
                return self.proxy.__getattr__(item)
            except Pyro4.errors.CommunicationError, Pyro4.errors.ConnectionClosedError:
                self.logger.warn("Connection lost, retrying")
                self.proxy._pyroReconnect()
                self.logger.info("Connection restored")

class Task:
    def __init__(self, type, data, **kwargs):
        self.id = -1#random.randint(0, 2 ** 31)
        self.data = data
        self.type = type
        self.owner = None
        self.params = kwargs

    def __getitem__(self, item):
        return self.params[item]

    def __repr__(self):
        return '<Task id=%d, owner=%s>' % (self.id, self.owner)