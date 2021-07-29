import numpy as np
from numpy import log, dot, e
from numpy.random import rand
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

# Static method
def sigmoid(z): return 1 / (1 + e ** (-z))

class LogisticRegression(BaseEstimator, ClassifierMixin):

    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate

    def cost_function(self, X, y, weights):
        z = dot(X, weights)
        predict_1 = y * log(sigmoid(z))
        predict_0 = (1 - y) * log(1 - sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)

    def accuracy(self, y, y_hat):
        acc = np.sum(y == y_hat) / len(y)
        return acc

    def fit(self, X, y, epochs=25):
        self.loss_ = []
        self.weights_ = rand(X.shape[1])
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        N = len(X)

        for _ in range(epochs):
            # Gradient Descent
            y_hat = sigmoid(dot(X, self.weights_))
            self.weights_ -= self.learning_rate * dot(X.T, y_hat - y) / N
            # Saving Progress
            self.loss_.append(self.cost_function(X, y, self.weights_))
        return self

    def predict(self, X):
        X = check_array(X)  # Validate the input
        check_is_fitted(self)  # Check to verify the Fit has been called
        # Predicting with sigmoid function
        z = dot(X, self.weights_)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in sigmoid(z)]

    def predict_proba(self, X):
        """This is a dummy method"""
        # TODO This is NOT an actual predict_proba. It's a demo.
        pred = np.random.rand(X.shape[0], self.classes_.size)
        return pred / np.sum(pred, axis=1)[:, np.newaxis]

    def score(self, X, y):
        """
        This method is called by GridSearch as a reference to tune the hyperparameters.
        :param X: data
        :param y: target
        :return: Mean accuracy of self.predict(X) wrt. y.
        """
        y_hat = self.predict(X)
        return self.accuracy(y, y_hat)


if __name__ == "__main__":
    # Hyperparameter Tunning
    X, y = make_blobs(n_samples=1000,
                      centers=2,
                      n_features=2,
                      random_state=1)

    grid_values = {
        "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.15, 0.2]
    }

    gs = GridSearchCV(LogisticRegression(),
                      grid_values)
    gs.fit(X, y)
    print("Best Param:", gs.best_params_)

    # Training model based on the best param values
    X_test, y_test = make_blobs(n_samples=10,
                                centers=2,
                                n_features=2,
                                random_state=1)

    lr = LogisticRegression(learning_rate=float(gs.best_params_['learning_rate']))
    lr.fit(X_test, y_test, epochs=25)

    # Evaluating model
    X, y = make_blobs(n_samples=5,
                      centers=2,
                      n_features=2,
                      random_state=1)

    y_hat = lr.predict(X)
    prob = lr.predict_proba(X)

    for i in range(len(X)):
        print("X=%s, y=%s, Predicted=%s, Prob Dist=%s" % (X[i], y[i], y_hat[i], prob[i]))
    print("Accuracy:", lr.accuracy(y, y_hat) * 100)
