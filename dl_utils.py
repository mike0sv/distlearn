import sys
import threading

import Pyro4

__author__ = 'Mike'

if sys.version_info[0] >= 3:
    # Python 3
    # noinspection PyShadowingBuiltins
    basestring = str
    # noinspection PyShadowingBuiltins
    unicode = str
    # noinspection PyShadowingBuiltins
    long = int


    def itervalues(d):
        return iter(d.values())


    def iteritems(d):
        return iter(d.items())
else:
    # Python 2
    # noinspection PyShadowingBuiltins
    range = xrange


    def itervalues(d):
        # noinspection PyCompatibility
        return d.itervalues()


    def iteritems(d):
        # noinspection PyCompatibility
        return d.iteritems()


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


# noinspection PyProtectedMember
class MasterWrapper:
    def __init__(self, uri, name, logger):
        self.name = name
        self.logger = logger
        self.proxy = Pyro4.core.Proxy(uri)

    def __getattr__(self, item):
        while True:
            try:
                return RemoteObjectWrapper(self.proxy.__getattr__(item), self.logger, self.proxy)
            except (Pyro4.errors.CommunicationError, Pyro4.errors.ConnectionClosedError):
                self.logger.warn("Connection lost, retrying")
                self.proxy._pyroReconnect()
                self.logger.info("Connection restored")


class RemoteObjectWrapper:
    def __init__(self, remote_object, logger, proxy):
        self.ro = remote_object
        self.logger = logger
        self.proxy = proxy

    def __call__(self, *args, **kwargs):
        while True:
            try:
                return self.ro(*args, **kwargs)
            except (Pyro4.errors.CommunicationError, Pyro4.errors.ConnectionClosedError) as e:
                self.logger.warn("Connection lost, retrying")
                self.proxy._pyroReconnect()
                self.logger.info("Connection restored")


class Task:
    def __init__(self, type, data, **kwargs):
        self.id = None  # random.randint(0, 2 ** 31)
        self.data = data
        self.type = type
        self.owner = None
        self.params = kwargs
        self.worker_id = None
        self.failed_by = dict()

    def __getitem__(self, item):
        return self.params[item]

    def __repr__(self):
        return '<Task id=%d, owner=%s>' % (self.id, self.owner)


"""
class bidict(dict):
    def __init__(self, *args, **kwargs):
        super(bidict, self).__init__(*args, **kwargs)
        self.inverse = {}
        for key, value in self.items():
            self.inverse.setdefault(value, []).append(key)

    def __setitem__(self, key, value):
        super(bidict, self).__setitem__(key, value)
        self.inverse.setdefault(value, []).append(key)

    def __delitem__(self, key):
        self.inverse.setdefault(self[key], []).remove(key)
        if self[key] in self.inverse and not self.inverse[self[key]]:
            del self.inverse[self[key]]
        super(bidict, self).__delitem__(key)
"""
