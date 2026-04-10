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

        return loss_history
    
    def fit_normal_descent_quantize_gradient_scaling(self, X, y, epochs, lr, total_bits=8, frac_bits=4, scaling_factor = 100):
        '''
        Performs Quantization-Aware Training (QAT) using Gradient Descent with Gradient Scaling.
        
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

            # Gradient Scaling: Scale gradients to mitigate quantization effects
            dw_scaled = dw * scaling_factor
            db_scaled = db * scaling_factor
            self.w -= lr * dw_scaled
            self.b -= lr * db_scaled

            # The QAT Step: Force weights into the fixed-point representation
            self.w = fixed_point_quantize(self.w, total_bits, frac_bits)
            self.b = fixed_point_quantize(self.b, total_bits, frac_bits)

            if np.linalg.norm(dw) < self.eps and abs(db) < self.eps:
                print("The loss converged at epoch:", epoch)
                break

            #calculating loss
            loss = np.mean(error**2)
            loss_history.append(loss)

        return loss_history
    


    def fit_with_shadow_weights(self, X, y, epochs, lr, total_bits=8, frac_bits=4):
        n_samples, n_features = X.shape
        
        #Initialize High-Precision "Shadow" Weights
        w_fp = np.zeros(n_features)
        b_fp = 0.0
        
        # Initialize the Quantized Weights (what we actually use)
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        loss_history = []
        lsb = 2**(-frac_bits)

        for epoch in range(epochs):
            # ALWAYS use the quantized weights for the forward pass 
            # to simulate the hardware error
            y_pred = X @ self.w + self.b 
            error = y_pred - y
            
            dw = (2 / n_samples) * X.T @ error
            db = (2 / n_samples) * np.sum(error)

            # Update the SHADOW weights (High Precision)
            w_fp -= lr * dw
            b_fp -= lr * db

            # Update the QUANTIZED weights by snapping the shadow weights
            self.w = fixed_point_quantize(w_fp, total_bits, frac_bits)
            self.b = fixed_point_quantize(b_fp, total_bits, frac_bits)

            loss = np.mean(error**2)
            loss_history.append(loss)
            
        return loss_history
    
    def fit_error_accumulation(self, X, y, epochs, lr, total_bits=8, frac_bits=4):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        loss_history = []
        error_acc_w = np.zeros_like(self.w)
        error_acc_b = 0
        lsb = 2**(-frac_bits)

        for epoch in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            
            dw = (2 / n_samples) * X.T @ error 
            db = (2 / n_samples) * np.sum(error)

            # calcualte step (error)
            w_step = -lr * dw
            b_step = -lr * db

            #accumulate the updates
            error_acc_w += w_step
            error_acc_b += b_step

            # detect which weights should be updated
            mask_up = error_acc_w > (0.5 * lsb)
            mask_down = error_acc_w < (-0.5 *lsb)
            
            # update the weights
            self.w[mask_up]+=lsb
            self.w[mask_down]-=lsb

            #substract from the accumulated error for weights
            error_acc_w[mask_up] -= lsb
            error_acc_w[mask_down] += lsb

            #same for the bias
            if error_acc_b > (0.5 * lsb):
                self.b += lsb
                error_acc_b -= lsb
            elif error_acc_b < (-0.5 * lsb):
                self.b -= lsb
                error_acc_b += lsb

            
            self.w = fixed_point_quantize(self.w, total_bits, frac_bits)
            self.b = fixed_point_quantize(self.b, total_bits, frac_bits)

            if np.linalg.norm(dw) < self.eps and abs(db) < self.eps:
                print("The loss converged at epoch:", epoch)
                break

            #calculating loss
            loss = np.mean(error**2)
            loss_history.append(loss)
        
        return loss_history
