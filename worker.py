from __future__ import print_function, with_statement, division
import Pyro4
__author__ = 'Mike'
import matplotlib.pyplot as plt
Pyro4.config.SERIALIZER = 'pickle'
Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')

def main():
    uri = 'PYRO:dispatcher@localhost:5555'
    proxy = Pyro4.core.Proxy(uri)
    X, Y = proxy.gimme_data()
    reg = proxy.gimme()
    #smth.do()
    X = X.reshape((1000, 1))
    Y = Y.reshape((1000, 1))
    print(X.shape, Y.shape)
    reg.fit(X, Y)
    plt.plot(X, reg.predict(X), color='black')
    plt.plot(X, Y, '.')
    plt.show()


if __name__ == '__main__':
    main()