import numpy as np
from src.quantization.quantize import fixed_point_quantize

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

        loss_history = []
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
            loss_history.append(loss)
            if epoch % 10 == 0:
                print("epoch:", epoch, "loss:", loss)

            #assign new weights and bias
            self.w = self.w - lr * dw
            self.b = self.b - lr * db
        return loss_history

    def fit_normal_descent_quantize(self, X, y, epochs, lr, total_bits=8, frac_bits=4):
        '''
        Performs Quantization-Aware Training (QAT) using Gradient Descent.
        
        This function simulates a model training directly on fixed-point embedded 
        hardware. It forces weights and biases to conform to a specific bit-width 
        and fractional precision after every update, allowing the optimizer to 
        attempt to compensate for quantization errors during the learning process.

        Parameters:
        -----------
        X : ndarray
            Training features.
        y : ndarray
            Training targets.
        epochs : int
            Number of iterations.
        lr : float
            Learning rate.
        total_bits : int
            Total word length (e.g., 8 or 16 bits).
        frac_bits : int
            Number of bits dedicated to the fractional part.

        Returns:
        --------
        loss_history : list
            A record of Mean Squared Error (MSE) at each epoch, capturing 
            convergence behavior under hardware constraints.
        '''
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        loss_history = []

        for epoch in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            
            dw = (2 / n_samples) * X.T @ error 
            db = (2 / n_samples) * np.sum(error)

            # Standard Update
            self.w -= lr * dw
            self.b -= lr * db

            # The QAT Step: Force weights into the fixed-point representation
            self.w = fixed_point_quantize(self.w, total_bits, frac_bits)
            self.b = fixed_point_quantize(self.b, total_bits, frac_bits)

            if np.linalg.norm(dw) < self.eps and abs(db) < self.eps:
                print("The loss converged at epoch:", epoch)
                break

            #calculating loss
            loss = np.mean(error**2)
            loss_history.append(loss)
            if epoch % 10 == 0:
                print("epoch:", epoch, "loss:", loss)

        return loss_history