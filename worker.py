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
        self.datasets = dict()
        self.id = name

    def check_data(self, client_id, task_id, name):
        if name in self.datasets:
            return True
        data = self.master.worker_get_data(name)
        if data is not None:
            self.datasets[name] = data
            return True
        else:
            self.master.worker_put_error(client_id, task_id, self.id,'Cannot load data %s' % name)
            return False

    def fit(self, task):
        data = self.datasets[task.data]
        train = data['data']
        target = data['target']
        if 'train' in task.params:
            train = train[task['train']]
            target = target[task['train']]
        clf = task['clf']
        clf.fit(train, target)
        self.master.worker_put_result(task.owner, task.id, clf)

    def fit_predict(self, task):
        data = self.datasets[task.data]
        train = data['data']
        target = data['target']
        if 'train' in task.params:
            train = train[task['train']]
            target = target[task['train']]
        clf = task['clf']
        clf.fit(train, target)
        test_data = data
        if 'test_data' in task.params:
            test_data = task['test_data']

        test = test_data['data']
        if 'test' in task.params:
            test = test[task['test']]

        if task['proba']:
            pred = clf.predict_proba(test)
        else:
            pred = clf.predict(test)

        if task['result'] == 'pred':
            self.master.worker_put_result(task.owner, task.id, pred)
        elif task['result'] == 'score':
            test_target = test_data['target']
            if 'test' in task.params:
                test_target = test_target[task['test']]
            score = task['scoring'](test_target, pred)
            self.master.worker_put_result(task.owner, task.id, score)

    def event_loop(self):
        try:
            while True:
                try:
                    logger.debug("Acquiring task")
                    task = self.master.worker_get_work()
                    if task is None:
                        logger.debug("No tasks available yet")
                        time.sleep(5)
                    else:
                        logger.info('Got %s' % task)
                        client_id, task_id = task.owner, task.id
                        if not self.check_data(client_id, task_id, task.data):
                            continue
                        if 'test_data' in task.params and not self.check_data(client_id, task_id, task['test_data']):
                            continue

                        if task.type == 'fit_predict':
                            self.fit_predict(task)
                        elif task.type == 'fit':
                            self.fit(task)

                        logger.info('Done')
                except Exception as e:
                    logger.error(e.message)
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