from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import pickle
import sys, os
import Queue
from random import randint, random

params = {'activation':'relu', 'solver':'lbfgs', 'batch_size':'auto',
          'learning_rate':'adaptive', 'learning_rate_init':0.001, 'power_t':0.5, 'max_iter':300, 'shuffle':True,
          'random_state':None, 'tol':0.1, 'verbose':False, 'warm_start':False, 'momentum':0.9,
          'nesterovs_momentum':True, 'early_stopping':False, 'validation_fraction':0.1, 'beta_1':0.9,
          'beta_2':0.999, 'epsilon':1e-08}

def output_test_predictions(X, y, X_test):
    for i in range(10):
        try:
            nn = pickle.load(open('../stored_neural_nets/nn' + str(i) + '.pkl', 'rb'))
        except IOError:
            break

        nn.fit(X, y)
        y_test = nn.predict(X_test)

        if not os.path.exists("../stored_neural_nets/submissions/"):
            os.makedirs("../stored_neural_nets/submissions/")  
        with open('../stored_neural_nets/submissions/subm' + str(i) + '.csv', 'w') as out:
            out.write("ID,Prediction\n")
            for i in range(1, 139):
                out.write(str(i) + "," + str(y_test[i - 1]) + "\n")

def try_params(X, y):
    lowest_mse = sys.maxint
    mlps = Queue.Queue(maxsize = 10) # Top 10.

    for i in range(30):
        if i % 5 == 0:
            print("NN Loop: " + str(i))

        # Monte Carlo the # of layers and neurons
        num_hidden_layers = randint(1, 20)
        layer_structure = [0] * num_hidden_layers
        for i in range(num_hidden_layers):
            layer_structure[i] = randint(1, 40)

        # Monte Carlo the value for alpha [0, 0.25)
        alpha = random() / 4

        # Train neural nets
        mlp = MLPRegressor(hidden_layer_sizes = tuple(layer_structure), alpha = alpha, **params)
        # mlp.fit(X, y)

        # print("Training set score: %f" % mlp.score(X, y))
        # print("Training set loss: %f" % mlp.loss_)

        # Cross-Validation
        scores = cross_val_score(mlp, X, y, scoring=make_scorer(mean_squared_error, greater_is_better=False), cv=3)
        # print("Accuracy: %0.2f (+/- %0.2f)" % (-scores.mean(), scores.std() * 2))

        if (-scores.mean() < lowest_mse):
            lowest_mse = -scores.mean()

            if mlps.qsize() >= 10:
                mlps.get()

            mlps.put(mlp)

    counter = 0
    while(not mlps.empty()):
        best_nn = mlps.get()
        if mlps.qsize() == 0:
            print("Best Neural Net Params:")
            print(best_nn.get_params())
        if not os.path.exists("../stored_neural_nets/"):
            os.makedirs("../stored_neural_nets/")    
        pickle.dump(best_nn, open('../stored_neural_nets/nn' + str(counter) + '.pkl', 'wb'))
        counter = counter + 1

    print("Lowest Error: " + str(lowest_mse))