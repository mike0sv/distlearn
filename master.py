from __future__ import print_function, with_statement, division

import argparse
import logging
import sys
import time
from threading import Lock

import numpy as np
from future.utils import iteritems

import dl_utils

try:
    # noinspection PyCompatibility
    import Queue as queue
except ImportError:
    # noinspection PyCompatibility
    import queue
__author__ = 'Mike'

dl_utils.Pyro4.config.SERIALIZER = 'pickle'
dl_utils.Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)

WORKER_FAIL_TIMEOUT = 5


# noinspection PyMethodMayBeStatic
class Master:
    def __init__(self, heartbeat_interval=5):
        self.workers = dict()
        self.heartbeat = dl_utils.RepeatedTimer(heartbeat_interval / 2., self._check_heartbeat)
        self.heartbeat.start()
        self.heartbeat_interval = heartbeat_interval
        self.task_aggregator = dl_utils.RepeatedTimer(1, self._aggregate_tasks)
        self.task_aggregator.start()
        self.work_queue = queue.Queue()
        self.clients = dict()
        self.lock_clients = Lock()
        self.datasets = dict()
        self.aggs = {'list': self._aggregate_tasks_list, 'hstack': self._aggregate_tasks_hstack,
                     'stacking': self._aggregate_tasks_stacking}

    def _aggregate_tasks_list(self, client, task_id, dep):
        # type: (Client, int, TaskDependency) -> None
        results = [client.results[child_id] for child_id in dep.deps]
        client.results[task_id] = results

    def _aggregate_tasks_hstack(self, client, task_id, dep):
        results = [client.results[child_id] for child_id in dep.deps]
        client.results[task_id] = np.hstack(results)

    def _aggregate_tasks_stacking(self, client, task_id, dep):
        if 'p3' not in dep.descr:
            with client.lock:
                fold_est_dict = dict(zip(dep.descr, map(lambda x: client.results[x], dep.deps)))
                est_count = len([0 for d in dep.descr if d.startswith('p2')])
                folds_count = int(len([0 for d in dep.descr if d.startswith('p1')]) / est_count)
                p3_train = np.vstack([np.vstack([fold_est_dict['p1_f%d_e%d' % (fold, n)] for n in range(est_count)]).T
                                      for fold in range(folds_count)])
                p3_test = np.vstack([fold_est_dict['p2_e%d' % n] for n in range(est_count)]).T
                data = {'data': p3_train, 'target': self.datasets[dep.args['data']]['target']}
                test_data = {'data': p3_test}
                estimator = dep.args['estimator']
                proba = dep.args['proba']
                result = dep.args['result']
                p3_task = client.create_task(type='fit_predict', result=result, data=data, test_data=test_data,
                                             estimator=estimator, proba=proba)
                client.task_dependencies[task_id].deps.append(p3_task.id)
                client.task_dependencies[task_id].descr.append('p3')
                self.work_queue.queue.append(p3_task)
            return False
        else:
            fold_est_dict = dict(zip(dep.descr, map(lambda x: client.results[x], dep.deps)))
            client.results[task_id] = fold_est_dict['p3']
            return True

    def _aggregate_tasks(self):
        for client in self.clients.values():
            done = list()
            for parent, dep in iteritems(client.task_dependencies):
                if all(map(lambda x: x in client.results, dep.deps)):
                    agg = self.aggs[dep.agg](client, parent, dep)
                    if agg is None or agg:
                        done.append(parent)
            map(client.task_dependencies.pop, done)

    def _check_heartbeat(self):
        clock = time.time()
        for name, worker in filter(lambda w: not w[1].dead, iteritems(self.workers)):
            if worker.last_heartbeat + self.heartbeat_interval < clock:
                logger.warn('Worker %s disconnected' % name)
                worker.dead = True
                # TODO remove rotting corpses

    def worker_send_heartbeat(self, worker_name):
        # TODO self.reanimate(workername)?
        self.workers[worker_name].last_heartbeat = time.time()
        self.workers[worker_name].dead = False

    def worker_register(self, worker_name):
        if worker_name in self.workers:
            logger.warn('Worker %s already connected' % worker_name)
        else:
            logger.info('Worker %s connected' % worker_name)
            self.workers[worker_name] = Worker(worker_name)

    def worker_get_work(self, worker_id):
        try:
            task = self.work_queue.get(timeout=.5)
            if worker_id in task.failed_by and time.time() - task.failed_by[worker_id] < WORKER_FAIL_TIMEOUT:
                self.work_queue.queue.appendleft(task)
                return None
            task.worker_id = worker_id
            return task
        except queue.Empty:
            return None

    def worker_put_result(self, client_id, task_id, result):
        try:
            logger.debug('Received result %d for %s' % (task_id, client_id))
            client = self.clients[client_id]
            client.results[task_id] = result
            del client.tasks[task_id]
        except:
            logger.error("Error", exc_info=True)

    def worker_put_error(self, client_id, task_id, worker_id, error):
        logger.info('Received error %s from worker %s in task %s' % (error, worker_id, task_id))
        try:
            client = self.clients[client_id]
            client.worker_errors.append((task_id, worker_id, error))
            task = client.tasks[task_id]
            task.failed_by[worker_id] = time.time()
            self.work_queue.queue.append(task)
        finally:
            # TODO
            pass

    def worker_get_data(self, name):
        try:
            return self.datasets[name]
        except KeyError:
            return None

    def _client_put_task(self, task):
        logger.debug('Adding task %d' % task.id)
        try:
            self.work_queue.queue.append(task)
        except Exception as e:
            print(e.message)

    def _check_params(self, task, params):
        if any(map(lambda x: x not in task.params, params)):
            raise AttributeError('Missing parameter; %s are needed' % ', '.join(params))

    def _cv_task(self, task):
        client = self.clients[task.owner]
        self._check_params(task, ['cv', 'estimator', 'scoring'])
        cv, estimator, scoring = task['cv'], task['estimator'], task['scoring']
        proba = 'proba' in task.params and task.params['proba']
        result = 'score' if 'result' not in task.params else task['result']
        agg_type = 'list' if result == 'score' else 'hstack'
        client.task_dependencies[task.id] = TaskDependency(task.id, agg_type)
        tasks = list()
        for train, test in cv:
            cv_task = client.create_task(type='fit_predict', data=task.data, result=result, scoring=scoring,
                                         estimator=estimator, proba=proba, train=train, test=test)
            client.task_dependencies[task.id].deps.append(cv_task.id)
            tasks.append(cv_task)
        map(self.work_queue.queue.append, tasks)
        return task.id

    def _stacking_task(self, task):
        client = self.clients[task.owner]
        self._check_params(task, ['cv', 'estimators', 'estimator', 'test_data'])

        cv, estimators, estimator, test_data = task['cv'], task['estimators'], task['estimator'], task['test_data']
        proba = 'proba' in task.params and task.params['proba']
        result = 'pred' if 'result' not in task.params else task['result']
        client.task_dependencies[task.id] = TaskDependency(task.id, 'stacking', estimator=estimator, proba=proba,
                                                           data=task.data, result=result, test_data=test_data)
        tasks = list()
        for fold, (train, test) in enumerate(cv):
            for n, est in enumerate(estimators):
                p1_task = client.create_task(type='fit_predict', result='pred', data=task.data,
                                             estimator=est, proba=proba, train=train, test=test)
                client.task_dependencies[task.id].deps.append(p1_task.id)
                client.task_dependencies[task.id].descr.append('p1_f%d_e%d' % (fold, n))
                tasks.append(p1_task)

        for n, est in enumerate(estimators):
            p2_task = client.create_task(type='fit_predict', result='pred', data=task.data, test_data=test_data,
                                         estimator=est, proba=proba)
            client.task_dependencies[task.id].deps.append(p2_task.id)
            client.task_dependencies[task.id].descr.append('p2_e%d' % n)
            tasks.append(p2_task)
        map(self.work_queue.queue.append, tasks)
        return task.id

    def client_put_task(self, task):
        if task.owner in self.clients:
            client = self.clients[task.owner]
            with client.lock:
                try:
                    client.add_task(task)
                    if task.type == 'cv':
                        return self._cv_task(task)
                    elif task.type == 'stacking':
                        return self._stacking_task(task)
                    else:
                        self.work_queue.queue.append(task)
                        return task.id
                except Exception:
                    logger.error('Error:', exc_info=True)
                    raise

    def client_put_data(self, name, data):
        logger.info('Got data "%s"' % name)
        self.datasets[name] = data

    def client_register(self, client_id):
        with self.lock_clients:
            if client_id in self.clients:
                logger.warn('Client %s already connected')
            else:
                logger.info('Client %s connected' % client_id)
                self.clients[client_id] = Client(client_id)

    def client_get_workers(self):
        return self.workers

    def client_get_errors(self, client_id, offset='new'):  # TODO offset? really?
        try:
            client = self.clients[client_id]
            next_error = 0
            if offset == 'new':
                next_error = client.next_error
            if type(offset) is int:
                next_error = offset
            client.next_error = len(client.worker_errors)
            return list(enumerate(client.worker_errors))[next_error:]
        finally:
            pass  # TODO

    def client_collect_task(self, client_id, task_id, timeout=None):
        logger.debug('Collecting result %d for %s' % (task_id, client_id))
        start = time.time()
        while task_id not in self.clients[client_id].results:
            if timeout is not None and time.time() - start() > timeout:
                return None
            time.sleep(.2)
        res = self.clients[client_id].results[task_id]
        #del self.clients[client_id].results[task_id]
        return res


class Worker:
    def __init__(self, worker_id):
        self.id = worker_id
        self.last_heartbeat = time.time()
        self.dead = False


class Client:
    def __init__(self, client_id):
        self.lock = Lock()
        self.id_lock = Lock()
        self.id = client_id
        self.results = dict()
        self.worker_errors = list()
        self.next_error = 0
        self.next_task_id = 0
        self.task_dependencies = dict()
        self.tasks = dict()

    def get_task_id(self):
        with self.id_lock:
            self.next_task_id += 1
            return self.next_task_id - 1

    def create_task(self, type, data, **kwargs):
        task = dl_utils.Task(type, data, **kwargs)
        task.owner = self.id
        task.id = self.get_task_id()
        self.tasks[task.id] = task
        return task

    def add_task(self, task):
        if task.id is None:
            task.id = self.get_task_id()
        self.tasks[task.id] = task


class TaskDependency:
    def __init__(self, parent, agg, deps=None, descr=None, **kwargs):
        self.deps = deps if deps is not None else list()
        self.descr = descr if descr is not None else list()
        self.parent = parent
        self.agg = agg
        self.args = kwargs


def start_master(host, port):
    daemon = dl_utils.Pyro4.core.Daemon(host, port)
    master = Master()
    uri = daemon.register(master, "master")

    logger.info("Master is running on " + str(uri))
    daemon.requestLoop()
    logger.info("Exiting")
    master.heartbeat.join()
    master.heartbeat.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', help='Master host', default='0.0.0.0')
    parser.add_argument('-p', '--port', help='Master port', default=5555, type=int)
    parser.add_argument('-v', '--verbosity', help='0=DEBUG 1=INFO 2=WARN 3=ERROR', default=1, type=int)
    args = parser.parse_args()
    levels = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARN, 3: logging.ERROR}
    logger.setLevel(levels[args.verbosity])
    start_master(args.host, args.port)


if __name__ == '__main__':
    main()
