from __future__ import print_function, with_statement, division

import argparse
import logging
import os
import socket
import time

from sklearn.utils import safe_indexing

from dl_utils import *

__author__ = 'Mike'

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
Pyro4.config.PICKLE_PROTOCOL_VERSION = 2

# Pyro4.config.COMMTIMEOUT = 1

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)

WORKERNAME = "Worker_%d@%s" % (os.getpid(), socket.gethostname())


# noinspection PyProtectedMember
class Heartbeat:
    def __init__(self, master, interval, name):
        self.master = Pyro4.core.Proxy(master._pyroUri)
        self.timer = RepeatedTimer(interval, self.send)
        self.timer.start()
        self.name = name

    def send(self):
        try:
            self.master.worker_send_heartbeat(self.name)
        except (Pyro4.errors.CommunicationError, Pyro4.errors.ConnectionClosedError):
            self.timer.pause()
            self.master._pyroReconnect()
            self.timer.resume()


# noinspection PyPep8Naming
class WorkerProxy:
    def __init__(self, name, host, port):
        logger.info("This is worker %s" % name)
        self.master = MasterWrapper("PYRO:master@%s:%d" % (host, port), name, logger)
        self.master.worker_register(name)
        logger.info("Connected to master")
        self.heartbeat_timer = Heartbeat(self.master.proxy, .5, name)
        self.datasets = dict()
        self.id = name

    def check_data(self, client_id, task_id, name):
        if not isinstance(name, str):
            return True
        if name in self.datasets:
            return True
        data = self.master.worker_get_data(name)
        if data is not None:
            self.datasets[name] = data
            return True
        else:
            self.master.worker_put_error(client_id, task_id, self.id, 'Cannot load data %s' % name)
            return False

    def get_data(self, task, need_test=True):
        # train, target, test, test_target = self.get_data(task)
        if isinstance(task.data, str):
            data = self.datasets[task.data]
        else:
            data = task.data

        train = data['data']
        target = data['target'] if 'target' in data else None

        if 'train' in task.params:
            train = safe_indexing(train, task['train'])
            if target is not None:
                target = safe_indexing(target, task['train'])
        if need_test:
            test_data = data
            if 'test_data' in task.params:
                if isinstance(task['test_data'], str):
                    test_data = self.datasets[task['test_data']]
                else:
                    test_data = task['test_data']

            test = test_data['data']
            if 'test' in task.params:
                test = safe_indexing(test, task['test'])

            test_target = test_data['target'] if 'target' in test_data else None
            if 'test' in task.params and test_target is not None:
                test_target = safe_indexing(test_target, task['test'])
        else:
            test = None
            test_target = None

        return train, target, test, test_target

    def fit(self, task):
        train, target, _, _ = self.get_data(task, False)
        estimator = task['estimator']
        estimator.fit(train, target)
        self.master.worker_put_result(task.owner, task.id, estimator)

    def fit_predict(self, task):
        train, target, test, test_target = self.get_data(task)

        estimator = task['estimator']
        estimator.fit(train, target)

        if 'proba' in task.params and task['proba']:
            pred = estimator.predict_proba(test)
        else:
            pred = estimator.predict(test)

        if task['result'] == 'pred':
            self.master.worker_put_result(task.owner, task.id, pred)
        elif task['result'] == 'score':
            score = task['scoring'](test_target, pred)
            self.master.worker_put_result(task.owner, task.id, score)

    def event_loop(self):
        while True:
            try:
                logger.debug("Acquiring task")
                task = self.master.worker_get_work(self.id)
                if task is None:
                    logger.debug("No tasks available yet")
                    time.sleep(5)
                else:
                    logger.info('Got %s' % task)
                    # raise AttributeError('lol')
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
                logger.error("This shitty log", exc_info=True)
                try:
                    client_id, task_id = task.owner, task.id
                    self.master.worker_put_error(client_id, task_id, self.id, 'Error: %s' % str(e.message))
                except AttributeError:
                    pass
                if isinstance(e, KeyboardInterrupt):
                    return
                    # raise

    def __enter__(self):
        return self

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        print('__exit__')
        self.heartbeat_timer.timer.stop()

    def close(self):
        print('__close__')
        self.heartbeat_timer.timer.stop()


def run_worker(name, host, port):
    with WorkerProxy(name, host, port) as worker:
        worker.event_loop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master', help='Master host', default='localhost')
    parser.add_argument('-p', '--port', help='Master port', default=5555, type=int)
    parser.add_argument('-v', '--verbosity', help='0=DEBUG 1=INFO 2=WARN 3=ERROR', default=1, type=int)
    args = parser.parse_args()
    levels = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN, 3: logging.ERROR}
    logger.setLevel(levels[args.verbosity])
    run_worker(WORKERNAME, args.master, args.port)
    logger.info('Exiting')


if __name__ == '__main__':
    main()
