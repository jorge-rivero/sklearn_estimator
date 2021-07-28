from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator
import numpy as np
import matplotlib.pyplot as plt

class LRScratch(BaseEstimator):
    def fit(self,X,y):
        X,y = check_X_y(X,y) #Checking if both X & y has correct shape
        self.n_features_in_ = X.shape[1] #Setting the number of features in Input data (new as per 0.24)
        self.Xb_ = np.c_[np.ones((X.shape[0],1)),X] #adding x0 = 1
        self.theta_ = np.linalg.inv(self.Xb_.T.dot(self.Xb_)).dot(self.Xb_.T).dot(y) # Solving theta using Normal Equation
        self.coef_ = self.theta_[1:]
        self.intercept_ = self.theta_[0]
        print("Training Completed")
        return self #Should Return Self : Mandatory

    def predict(self,X_test):
        check_is_fitted(self) # Check to verify the Fit has been called
        X_test = check_array(X_test) # Validate the input
        return X_test@self.coef_+self.intercept_

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100,1)
    y = 4 + 3*X + np.random.randn(100,1)
    plt.plot(X,y,"o")
    #plt.show()
    lr = LRScratch()
    lr.fit(X,y)
    print(lr.theta_)
    print(lr.predict([[5]]))
    
    #LR = LRScratch()
    #check_estimator(LR)


