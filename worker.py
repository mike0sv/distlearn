from __future__ import print_function, with_statement, division
import Pyro4
import sys
import os
import socket
import logging
import time
import Queue as queue
import matplotlib.pyplot as plt
__author__ = 'Mike'
from dl_utils import *

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
#Pyro4.config.COMMTIMEOUT = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)

WORKERNAME = "Worker_%d@%s" % (os.getpid(), socket.gethostname())

class Heartbeat():
    def __init__(self, master, interval):
            self.master = Pyro4.core.Proxy(master._pyroUri)
            self.timer = RepeatedTimer(interval, self.send)
            self.timer.start()
    def send(self):
        try:
            self.master.worker_send_heartbeat(WORKERNAME)
        except:
            self.timer.pause()
            self.master._pyroReconnect()
            self.timer.resume()


class WorkerProxy:
    def __init__(self, name, addr):
        logger.info("This is worker %s" % name)
        self.master = MasterWrapper("PYRO:master@" + addr, name, logger)
        self.master.worker_register(name)
        self.heartbeat_timer = Heartbeat(self.master.proxy, .5)

    def event_loop(self):
        try:
            while True:
                logger.debug("Acquiring work")
                work = self.master.worker_get_work()
                if work is None:
                    logger.debug("No work available yet")
                    time.sleep(5)
                else:
                    logger.info('Got %s' % work)
                    X, Y = work.data
                    work.clf.fit(X, Y)
                    logger.info("Returning result")
                    import temp
                    print(temp.getsize(work.clf))
                    self.master.worker_put_result(work.id, work.clf)
                    logger.info('Done')
        except Exception:
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.heartbeat_timer.timer.stop()

    def close(self):
        self.heartbeat_timer.timer.stop()


def main():
    uri = '%s:%d' % (sys.argv[1], 5555)
    #proxy = Pyro4.core.Proxy(uri)
    with WorkerProxy(WORKERNAME, uri) as worker:
        worker.event_loop()
    logger.info('Exiting')


if __name__ == '__main__':
    main()