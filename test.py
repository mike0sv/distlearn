from __future__ import print_function
from __future__ import print_function

import logging
import time

import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss

import client


def profile(func):
    def wrap(*args, **kwargs):
        started_at = time.time()
        result = func(*args, **kwargs)
        logging.info(time.time() - started_at)
        return result

    return wrap


@profile
def main():
    print('Connecting')
    dc = client.Client(client.CLIENTNAME, 'localhost', 5555)

    print('Reading data')
    data = pd.read_csv('./data/holdout.csv')
    target = data['target'].values
    data.drop('target', inplace=True, axis=1)
    data.drop(filter(lambda c: data[c].dtype == 'O', data.columns), inplace=True, axis=1)
    data.fillna(-999, inplace=True)
    print('Sending data')
    dc.send_data('bnp', {'data': data.as_matrix(), 'target': target})
    print('Cross val')
    clf = ExtraTreesClassifier(n_estimators=1000, max_features=60, criterion='entropy', min_samples_split=4,
                               max_depth=40, min_samples_leaf=2, n_jobs=-1)
    #clf.fit(data, target)
    scores = []
    scores = dc.cross_validate('bnp', clf, log_loss, StratifiedKFold(target, 10), async=False)
    print(dc.master.client_get_errors(dc.id))
    print(scores)
    print(sum(scores) / len(scores))


if __name__ == '__main__':
    main()
