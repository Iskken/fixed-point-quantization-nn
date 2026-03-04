import numpy as np

class LinearRegression():
    def __init__(self):
        self.w = None
        self.b = None
        self.eps = 1e-6

    #Outputs the predicted values in a matrix
    def predict(self, X):
        return X @ self.w + self.b
    
    def fit_gradient_descent(self, X, y, epochs, lr):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for epoch in range(epochs):
            #predict with the current weights and bias
            y_pred = self.predict(X)

            #calculate the weight and bias gradient
            error = y_pred - y
            #X transpose is taken since it is originally n * 1 along with (y_pred - y), 
            #so we need to make it (1 * n) dim-s to obtain 1x1 in the end
            dw = (2 / n_samples) * X.T @ (error) 
            db = (2/n_samples) * np.sum(error)

            if np.linalg.norm(dw) < self.eps and abs(db) < self.eps:
                print("The loss converged at epoch:", epoch)
                break

            #calculating loss
            loss = np.mean(error**2)
            if epoch % 10 == 0:
                print("epoch:", epoch, "loss:", loss)

            #assign new weights and bias
            self.w = self.w - lr * dw
            self.b = self.b - lr * db