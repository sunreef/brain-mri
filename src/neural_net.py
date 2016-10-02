from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import pickle
import sys
from random import randint

params = {'activation':'relu', 'solver':'adam', 'alpha':0.0001, 'batch_size':'auto',
          'learning_rate':'constant', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':200, 'shuffle':True,
          'random_state':None, 'tol':0.0001, 'verbose':False, 'warm_start':False, 'momentum':0.9,
          'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9,
          'beta_2':0.999, 'epsilon':1e-08}


def try_params(X, y):
    lowest_mse = sys.maxint

    # Monte Carlo the # of layers and neurons
    num_hidden_layers = randint(1, 8)
    layer_structure = [0] * num_hidden_layers
    for i in range(num_hidden_layers):
        layer_structure[i] = randint(1, 40)

    for i in range(50):
        if i % 5 == 0:
            print("Loop: " + str(i))
        # Train neural nets
        mlp = MLPRegressor(hidden_layer_sizes = tuple(layer_structure), **params)
        # mlp.fit(X, y)

        # print("Training set score: %f" % mlp.score(X, y))
        # print("Training set loss: %f" % mlp.loss_)

        # Cross-Validation
        scores = cross_val_score(mlp, X, y, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=3)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (-scores.mean(), scores.std() * 2))

        if (-scores.mean() < lowest_mse):
            lowest_mse = -scores.mean()
            pickle.dump(mlp, open('nn.pkl', 'wb'))

    nn = pickle.load(open('nn.pkl', 'rb'))
    print("Lowest Error: " + str(lowest_mse))
    print(nn.get_params())