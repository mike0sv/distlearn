from __future__ import print_function, with_statement, division
from future.utils import iteritems
import time
import logging
import sys
from threading import Lock
import dl_utils

try:
    # noinspection PyCompatibility
    import Queue as queue
except ImportError:
    # noinspection PyCompatibility
    import queue
__author__ = 'Mike'

#dl_utils.Pyro4.config.SERIALIZER = 'pickle'
#dl_utils.Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)


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

    def _aggregate_tasks_list(self, client, task_id, deps):
        # type: (Client, int, TaskDependency) -> None
        results = [client.results[child_id] for child_id in deps]
        client.results[task_id] = results

    def _aggregate_tasks(self):
        for client in self.clients.values():
            done = list()
            for parent, dep in iteritems(client.task_dependencies):
                if all(map(lambda x: x in client.results, dep.deps)):
                    if dep.agg == 'list':
                        self._aggregate_tasks_list(client, parent, dep.deps)
                    done.append(parent)
            map(client.task_dependencies.pop, done)

    def _check_heartbeat(self):
        clock = time.time()
        for name, worker in filter(lambda w: not w[1].dead, iter(self.workers.items())):
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

    def worker_get_work(self):
        try:
            return self.work_queue.get(timeout=.5)
        except queue.Empty:
            return None

    def worker_put_result(self, client_id, task_id, result):
        logger.debug('Received result %d for %s' % (task_id, client_id))
        self.clients[client_id].results[task_id] = result

    def worker_put_error(self, client_id, task_id, worker_id, error):
        logger.info('Received error %s from worker %s in task %s' % (error, worker_id, task_id))
        try:
            self.clients[client_id].worker_errors.append((task_id, worker_id, error))
        except:
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

    def client_put_task(self, task):
        if task.owner in self.clients:
            client = self.clients[task.owner]
            client.lock.acquire()
            try:
                task.id = client.next_task_id
                client.next_task_id += 1
                if task.type == 'cv':
                    if any(map(lambda x: x not in task.params, ['cv', 'clf', 'scoring'])):
                        raise AttributeError('Missing parameter; cv, clf, scoring are needed')
                    cv, clf, scoring = task['cv'], task['clf'], task['scoring']
                    proba = 'proba' in task.params and task.params['proba']
                    client.task_dependencies[task.id] = TaskDependency(task.id, 'list')
                    tasks = list()
                    for train, test in cv:
                        task_id = client.next_task_id #TODO
                        client.next_task_id += 1
                        cv_task = dl_utils.Task(type='fit_predict', data=task.data, result='score', scoring=scoring,
                                                clf=clf, proba=proba, train=train, test=test)
                        cv_task.owner = task.owner
                        cv_task.id = task_id
                        client.task_dependencies[task.id].deps.append(task_id)
                        tasks.append(cv_task)
                    map(self.work_queue.queue.append, tasks)
                    return task.id
                else:
                    self.work_queue.queue.append(task)
                    return task.id
            finally:
                client.lock.release()

    def client_put_data(self, name, data):
        self.datasets[name] = data

    def client_register(self, client_id):
        self.lock_clients.acquire()
        try:
            if client_id in self.clients:
                logger.warn('Client %s already connected')
            else:
                logger.info('Client %s connected' % client_id)
                self.clients[client_id] = Client(client_id)
        finally:
            self.lock_clients.release()

    def client_get_workers(self):
        return self.workers

    def client_get_errors(self, client_id, offset='new'):  # TODO offset? really?
        try:
            next_error = 0
            if offset == 'new':
                next_error = self.clients[client_id].next_error
            if type(offset) is int:
                next_error = offset
            if next_error < self.clients[client_id].next_error:
                self.clients[client_id].next_error = next_error
            return self.clients[client_id].worker_errors[next_error:]
        except:
            pass  # TODO

    def client_collect_task(self, client_id, task_id, timeout=None):
        logger.debug('Collecting result %d' % task_id)
        start = time.time()
        while task_id not in self.clients[client_id].results:
            if timeout is not None and time.time() - start() > timeout:
                return None
            time.sleep(.2)
        res = self.clients[client_id].results[task_id]
        del self.clients[client_id].results[task_id]
        return res


class Worker:
    def __init__(self, id):
        self.id = id
        self.last_heartbeat = time.time()
        self.dead = False


class Client:
    def __init__(self, id):
        self.lock = Lock()
        self.id = id
        self.results = dict()
        self.worker_errors = list()
        self.next_error = 0
        self.next_task_id = 0
        self.task_dependencies = dict()

    def get_task_id(self):
        self.next_task_id += 1
        return self.next_task_id - 1


class TaskDependency:
    def __init__(self, parent, agg, deps=None):
        if deps is None:
            self.deps = list()
        else:
            self.deps = deps
        self.parent = parent
        self.agg = agg


def main():
    host = '0.0.0.0'
    port = 5555

    daemon = dl_utils.Pyro4.core.Daemon(host, port)
    master = Master()
    uri = daemon.register(master, "master")

    logger.info("Master is running on " + str(uri))
    daemon.requestLoop()


if __name__ == '__main__':
    main()
