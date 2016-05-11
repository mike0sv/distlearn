import logging
import os
import socket

from dl_utils import *

__author__ = 'Mike'

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s'))
logger.addHandler(log_handler)

CLIENTNAME = "Client_%d@%s" % (os.getpid(), socket.gethostname())


class Client:
    def __init__(self, name, host='localhost', port=5555):
        address = '%s:%d' % (host, port)
        self.id = name
        logger.info("This is client %s" % name)
        self.master = MasterWrapper("PYRO:master@" + address, name, logger)
        self.master.client_register(name)
        logger.info("Connected to master")

    def send_task(self, task, async=True):
        task.owner = self.id
        task.id = None
        task.id = self.master.client_put_task(task)
        if async:
            return task.id
        else:
            return self.collect_task(task.id)

    def send_data(self, data_name, data):
        if not isinstance(data, dict) or 'data' not in data:
            raise AttributeError('Wrong data format')
        self.master.client_put_data(data_name, data)

    def cross_validate(self, description, data, estimator, scoring, cv, async=True):
        task = Task(data=data, description=description, estimator=estimator, scoring=scoring, cv=cv, type='cv')
        return self.send_task(task, async)

    def stacking(self, description, data, test_data, estimators, estimator, cv, result='pred', async=True):
        task = Task(data=data, description=description, test_data=test_data, estimators=estimators, estimator=estimator,
                    result=result, cv=cv, type='stacking')
        return self.send_task(task, async)

    def collect_task(self, task_id):
        result = self.master.client_collect_task(self.id, task_id)
        return result

    def check_errors(self):
        return self.master.client_get_errors(self.id)

    def cancel_task(self, task_id):
        return self.master.client_cancel_task(self.id, task_id)

    def list_tasks(self, auto=False, failed=None, completed=None, canceled=None):
        return self.master.client_list_tasks(self.id, auto, failed, completed, canceled)

    def get_task(self, task_id):
        return self.master.client_get_task(self.id, task_id)


def test1(client):
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np

    dataX = np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataX.sort()
    dataX = dataX.reshape((1000, 1))
    dataY = 5 * np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataY.sort()
    dataY = dataY.reshape((1000, 1))
    client.send_data('data1', {'data': dataX, 'target': dataY})

    id1 = client.send_task(Task(type='fit', data='data1',
                                estimator=RandomForestRegressor(n_estimators=1000, max_depth=1)))
    id2 = client.send_task(Task(type='fit', data='data1',
                                estimator=RandomForestRegressor(n_estimators=1000, max_depth=100)))
    res1 = client.collect_task(id1)
    res2 = client.collect_task(id2)
    print('done')
    import matplotlib.pyplot as plt

    plt.plot(dataX, res1.predict(dataX), color='green')
    plt.plot(dataX, res2.predict(dataX), color='red')
    plt.plot(dataX, dataY, '.', alpha=.1)
    plt.show()


def test2(client):
    """

    :type client: Client
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    from sklearn.cross_validation import ShuffleSplit
    import numpy as np

    dataX = np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataX.sort()
    dataX = dataX.reshape((1000, 1))
    dataY = 5 * np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataY.sort()
    dataY = dataY.reshape((1000, 1))
    client.send_data('data1', {'data': dataX, 'target': dataY})

    id1 = client.cross_validate('data1', RandomForestRegressor(n_estimators=1000, max_depth=100),
                                mean_squared_error, ShuffleSplit(1000, 5, .2))

    res1 = client.collect_task(id1)
    print('done')
    print(res1)


def test3(client):
    """

    :type client: Client
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.cross_validation import KFold
    import numpy as np

    dataX = np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataX.sort()
    dataX = dataX.reshape((1000, 1))
    dataY = 5 * np.ones(1000) + np.random.uniform(0, 1, 1000)
    dataY.sort()
    dataY = dataY.reshape((1000, 1))
    client.send_data('data1', {'data': dataX, 'target': dataY})

    id1 = client.stacking('stack task1', 'data1', 'data1', [RandomForestRegressor(n_estimators=1000, max_depth=100),
                                                            RandomForestRegressor(n_estimators=1000, max_depth=2)],
                          LinearRegression(), KFold(1000, 3))

    res1 = client.collect_task(id1)
    print('done')
    print(res1)

    import matplotlib.pyplot as plt

    plt.plot(dataX, res1, color='green')
    plt.plot(dataX, dataY, '.', alpha=.1)
    plt.show()


def main():
    client = Client(CLIENTNAME, sys.argv[1], 5555)
    # test1(client)
    # test2(client)
    test3(client)


if __name__ == '__main__':
    main()
