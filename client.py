import os
import socket
import logging
import sys

__author__ = 'Mike'
from dl_utils import *

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)

CLIENTNAME = "Client_%d@%s" % (os.getpid(), socket.gethostname())


class Client:
    def __init__(self, name, host, port):
        address = '%s:%d' % (host, port)
        self.id = name
        logger.info("This is client %s" % name)
        self.master = MasterWrapper("PYRO:master@" + address, name, logger)
        self.master.client_register(name)

    def send_task(self, task, async=True):
        task.owner = self.id
        task.id = self.master.client_put_task(task)
        if async:
            return task.id
        else:
            return self.collect_task(task.id)

    def send_data(self, data_name, data):
        self.master.client_put_data(data_name, data)

    def cross_validate(self, data, clf, scoring, cv, async=True):
        task = Task(data=data, clf=clf, scoring=scoring, cv=cv, type='cv')
        task.owner = self.id
        task.id = self.master.client_put_task(task)

    def collect_task(self, task_id):
        result = self.master.client_collect_task(self.id, task_id)
        return result


def test1(client):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    dataX = np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataX.sort()
    dataX = dataX.reshape((1000, 1))
    dataY = 5 * np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataY.sort()
    dataY = dataY.reshape((1000, 1))
    client.send_data('data1', {'data':dataX, 'target':dataY})

    id1 = client.send_task(Task(type='fit', data='data1', clf=RandomForestRegressor(n_estimators=1000, max_depth=1)))
    id2 = client.send_task(Task(type='fit', data='data1', clf=RandomForestRegressor(n_estimators=1000, max_depth=100)))
    res1 = client.collect_task(id1)
    res2 = client.collect_task(id2)
    print('done')
    import matplotlib.pyplot as plt

    plt.plot(dataX, res1.predict(dataX), color='green')
    plt.plot(dataX, res2.predict(dataX), color='red')
    plt.plot(dataX, dataY, '.', alpha=.1)
    plt.show()


def main():
    client = Client(CLIENTNAME, sys.argv[1], 5555)
    test1(client)


if __name__ == '__main__':
    main()