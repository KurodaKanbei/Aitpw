import xgboost
from sklearn.model_selection import train_test_split
import numpy as np
import os
import argparse

#   general arguments
parser = argparse.ArgumentParser(description='use xgboost to train a text classifier and do a predict')
parser.add_argument('-fsave-model', type=str, default='./models/new%d.model')
parser.add_argument('-auto', action='store_true', default=False)
parser.add_argument('-cv', action='store_true', default=False)
parser.add_argument('-predict-only', action='store_true', default=False)

#   cross validataion arguments
# parser.add_argument('-nfold', type=int, default=4)

#   train arguments
parser.add_argument('-snapshot', type=str, default=None)
parser.add_argument('-num-round', type=int, default=100)
parser.add_argument('-epoch', type=int, default=50)
parser.add_argument('-max-depth', type=int, default=8)
parser.add_argument('-subsample', type=float, default=0.600)
#parser.add_argument('-early-stopping-rounds', type=int, default=3)
parser.add_argument('-colsample-bytree', type=float, default=0.7)
parser.add_argument('-colsample-bylevel', type=float, default=0.7)
parser.add_argument('-eta', type=float, default=0.3)            #   step size shrinkage used in update to prevents overfitting
#parser.add_argument('-max-leaf-nodes', type=int, default=80)
parser.add_argument('-scale-pos-weight', type=float, default=12.0)
parser.add_argument('-silent', type=int, default=0)
parser.add_argument('-objective', type=str, default='binary:logistic')
parser.add_argument('-nthread', type=int, default=4)
parser.add_argument('-eval-metric', type=str, default='auc')

args = parser.parse_args()

if __name__ == '__main__':

    source = 'features.npy'

    m = int(125e4)
    n_pos = int(1e6)
    n_neg = int(1e6)
    n_full = n_pos + n_neg
    n_test = int(1e4)

    D = np.load(os.path.join('data', source))

    X_pos = D[:n_pos]
    X_neg = D[m: m + n_neg]
    X = np.concatenate((X_pos, X_neg), axis=0)
    print('load successfully...')

    Y = np.asarray([1] * n_pos + [0] * n_neg)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    D_train = xgboost.DMatrix(X_train, label=Y_train)

    D_test = xgboost.DMatrix(X_test, label=Y_test)

    print('train model...')
    eval_list = [(D_train, 'train'), (D_test, 'test')]

    bst = xgboost.train(args.__dict__, D_train, args.num_round, eval_list)

    print('save model...')
    cur = 0
    while os.path.exists(os.path.join('models', args.fsave_model % cur)):
        cur += 1
    fname = os.path.join(args.fsave_model % cur)
    bst.save_model(fname)
    print('%s saved.' % (fname))

