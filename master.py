from __future__ import print_function, with_statement, division
import Pyro4
import time
import logging
import sys
try:
    import Queue as queue
except:
    import queue
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor
__author__ = 'Mike'
from dl_utils import *

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)

class Master:
    def __init__(self, heartbeat_interval = 5):
        self.workers = dict()
        self.heartbeat = RepeatedTimer(heartbeat_interval / 2., self._check_heartbeat)
        self.heartbeat.start()
        self.heartbeat_interval = heartbeat_interval
        self.work_queue = queue.Queue()
        self.results = dict()

    def _check_heartbeat(self):
        clock = time.time()
        for name, worker in filter(lambda w: not w[1].dead, self.workers.iteritems()):
            if worker.last_heartbeat + self.heartbeat_interval < clock:
                logger.warn('Worker %s disconnected' % name)
                worker.dead = True
        #TODO remove rotting corpses

    def worker_send_heartbeat(self, worker_name):
        #TODO self.reanimate(workername)?
        self.workers[worker_name].last_heartbeat = time.time()
        self.workers[worker_name].dead = False

    def worker_register(self, worker_name):
        if worker_name in self.workers:
            logger.warn('Worker %s already connected' % worker_name)
        else:
            logger.info('Worker %s connected' % worker_name)
            self.workers[worker_name] = Worker(worker_name)

    def worker_get_work(self):
        try:
            return self.work_queue.get(timeout=.5)
        except queue.Empty:
            return None

    def worker_put_result(self, task_id, result):
        logger.debug('Received result %d' % task_id)
        self.results[task_id] = result

    def client_put_task(self, task):
        logger.debug('Adding task %d' % task.id)
        try:
            self.work_queue.queue.append(task)
        except Exception as e:
            print(e.message)

    def client_register(self, name):
        logger.info('Client %s connected' % name)

    def client_collect_task(self, id, timeout=None):
        logger.debug('Collecting result %d' % id)
        start = time.time()
        while id not in self.results:
            if timeout is not None and time.time() - start() > timeout:
                return None
        res = self.results[id]
        del self.results[id]
        return res


class Worker:
    def __init__(self, name):
        self.name = name
        self.last_heartbeat = time.time()
        self.dead = False

def main():
    host = '0.0.0.0'
    port = 5555

    daemon = Pyro4.core.Daemon(host, port)
    master = Master()
    uri = daemon.register(master, "master")

    logger.info("Master is running on " + str(uri))
    daemon.requestLoop()

if __name__ == '__main__':
    main()