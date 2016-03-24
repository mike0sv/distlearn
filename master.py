from __future__ import print_function, with_statement, division
import Pyro4
from sklearn.linear_model import LinearRegression as LR
from sklearn.ensemble import RandomForestRegressor
__author__ = 'Mike'

Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')



class Master:
    def __init__(self):
        self.buffer = list()
    def lol(self):
        print('lol1')
    def gimme(self):
        return LR()
        #return RandomForestRegressor(n_estimators=10000)
    def gimme_data(self):
        import numpy as np
        dataX = np.ones(1000) + np.random.uniform(0, 1, 1000)
        dataX.sort()
        dataY = 5 * np.ones(1000) + np.random.uniform(0, 1, 1000)
        dataY.sort()
        return dataX, np.log(dataY)
    def ou(self):
        return self.buffer

def main():
    host = 'localhost'
    port = 5555

    daemon = Pyro4.core.Daemon(host, port)
    dispatcher = Master()
    uri = daemon.register(dispatcher, "dispatcher")

    print("Dispatcher is running: " + str(uri))
    daemon.requestLoop()

if __name__ == '__main__':
    main()