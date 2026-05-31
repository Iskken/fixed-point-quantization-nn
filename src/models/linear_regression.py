import math

import numpy as np
from src.quantization.quantize import fixed_point_quantize

class LinearRegression():
    def __init__(self):
        self.w = None
        self.b = None
        self.eps = 1e-6

    #helper function for calculating the gradient and loss based on the given loss_function
    def _calculate_grad_and_loss(self, X, n_samples, error, loss_func = "MSE", delta = 2.0):
        if loss_func == "MSE":
            loss = np.mean(error**2)
            #X transpose is taken since it is originally n * 1 along with (y_pred - y), 
            #so we need to make it (1 * n) dim-s to obtain 1x1 in the end
            dw = (2 / n_samples) * X.T @ error
            db = (2 / n_samples) * np.sum(error)

        elif loss_func == "MAE":
            loss = np.mean(np.abs(error))
            # The derivative of absolute value is the 'sign' (+1 or -1)
            dw = (1 / n_samples) * X.T @ np.sign(error)
            db = (1 / n_samples) * np.sum(np.sign(error))

        elif loss_func == "Huber":
            # Create a mask to separate small errors from huge outliers
            is_small_error = np.abs(error) <= delta
            
            # Calculate Loss
            squared_loss = 0.5 * error**2
            linear_loss = delta * (np.abs(error) - 0.5 * delta)
            loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))
            
            # Calculate Gradients
            # If small error: gradient is just the error. If large: gradient is delta * sign(error)
            grad_error = np.where(is_small_error, error, delta * np.sign(error))
            dw = (1 / n_samples) * X.T @ grad_error
            db = (1 / n_samples) * np.sum(grad_error)
        else:
            raise ValueError("Unknown loss function!")
        
        return dw, db, loss
    

    #Outputs the predicted values in a matrix
    def predict(self, X):
        return X @ self.w + self.b
    
    def fit_gradient_descent(self, X, y, epochs, lr, loss_func = "MSE"):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        loss_history = []
        for epoch in range(epochs):
            #predict with the current weights and bias
            y_pred = self.predict(X)

            #calculate the weight and bias gradient
            error = y_pred - y
            
            dw, db, loss = self._calculate_grad_and_loss(X, n_samples, error, loss_func=loss_func)


            if np.linalg.norm(dw) < self.eps and abs(db) < self.eps:
                print("The loss converged at epoch:", epoch)
                break

            #calculating loss
            loss_history.append(loss)

            #assign new weights and bias
            self.w = self.w - lr * dw
            self.b = self.b - lr * db
        return loss_history

    def fit_normal_descent_quantize(self, X, y, epochs, lr, total_bits=8, frac_bits=4, loss_func="MSE"):
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
        loss_func : str
            The type of lose function that should be used
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
            
            dw, db, loss = self._calculate_grad_and_loss(X, n_samples, error, loss_func=loss_func)

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
            loss_history.append(loss)

        return loss_history
    
    def fit_normal_descent_quantize_gradient_scaling(self, X, y, epochs, lr, total_bits=8, frac_bits=4, scaling_factor = 100, loss_func = "MSE"):
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
        loss_func : str
            The type of lose function that should be used

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
            
            dw, db, loss = self._calculate_grad_and_loss(X, n_samples, error, loss_func=loss_func)

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
            loss_history.append(loss)

        return loss_history
    
    
    
    def fit_error_accumulation(self, X, y, epochs, lr, total_bits=8, frac_bits=4, loss_func="MSE"):
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0
        loss_history = []
        
    
        error_acc_w = np.zeros_like(self.w)
        error_acc_b = 0.0
        lsb = 2**(-frac_bits)

        for epoch in range(epochs):
            y_pred = self.predict(X)
            error = y_pred - y
            
            dw, db, loss = self._calculate_grad_and_loss(X, n_samples, error, loss_func=loss_func)

            # step
            w_step = -lr * dw
            b_step = -lr * db

            # ideal udpate
            w_temp = self.w + w_step
            b_temp = self.b + b_step

            # quantized update
            w_quan = fixed_point_quantize(w_temp, total_bits, frac_bits)
            b_quan = fixed_point_quantize(b_temp, total_bits, frac_bits)

            # quantization error
            w_quant_error = w_temp - w_quan
            b_quant_error = b_temp - b_quan

            # adding the quantization error to accumulator
            error_acc_w += w_quant_error
            error_acc_b += b_quant_error

            # appying hardware allowed weights to physical weights
            self.w = w_quan
            self.b = b_quan

            # Did the chopped-off errors build up to a full LSB?
            mask_up = error_acc_w > (0.5 * lsb)
            mask_down = error_acc_w < (-0.5 * lsb)
            
            # Compensate: Force the weights up/down
            self.w[mask_up] += lsb
            self.w[mask_down] -= lsb

            # empty the accumulators to preven double counting
            error_acc_w[mask_up] -= lsb
            error_acc_w[mask_down] += lsb

            # do same for bias
            if error_acc_b > (0.5 * lsb):
                self.b += lsb
                error_acc_b -= lsb
            elif error_acc_b < (-0.5 * lsb):
                self.b -= lsb
                error_acc_b += lsb

            # Safety Net: Ensure the manual LSB bumps didn't create floating point drift
            self.w = fixed_point_quantize(self.w, total_bits, frac_bits)
            self.b = fixed_point_quantize(self.b, total_bits, frac_bits)

            loss_history.append(loss)

        return loss_history


    def compute_zoo_cost(self, y_pred, y, loss_func="MSE", delta=2.0):
        """
        Hardware-friendly cost function. 
        Uses SUM instead of MEAN to avoid floating-point division.
        """
        error = y_pred - y

        if loss_func == "MSE":
            return np.sum(error**2) # SSE
        elif loss_func == "MAE": 
            return np.sum(np.abs(error)) # SAD (Sum of Absolute Differences)
        elif loss_func == "Huber":
            is_small = np.abs(error) <= delta
            squared = 0.5 * error**2
            linear = delta * (np.abs(error) - 0.5 * delta)
            return np.sum(np.where(is_small, squared, linear))
        else:
            raise ValueError("Unknown loss function")

    def fit_zoo(
        self,
        X,
        y,
        epochs=100,
        lr_shift=6,
        batch_size=32,
        sub_batch_size=8,
        total_bits=8,
        frac_bits=4,
        delta = 0.2,
        loss_func="MSE"
    ):
        lr = 2**(-lr_shift)

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        lsb = 2 ** (-frac_bits)
        loss_history = []
        loss_history_hardware = []

        for epoch in range(epochs):

            #Shuffling the data: Model doesn't memorize the order of the data. 
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]

            ## Process data in mini batches(32 samples)
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                current_batch_size = X_batch.shape[0]

                
                num_sub_batches = max(1, current_batch_size // sub_batch_size)

                #sub batches for multiple gradient estimates
                sub_batches = np.array_split(np.arange(current_batch_size), num_sub_batches)

                grad_w_total = np.zeros_like(self.w)
                grad_b_total = 0.0

                #calculate the gradient for each sub-batch
                for sub_idx in sub_batches:
                    X_sub = X_batch[sub_idx]
                    y_sub = y_batch[sub_idx]

                    ## set the random direction for each weight and bias
                    direction_w = np.random.choice([-1, 1], size=self.w.shape)
                    direction_b = np.random.choice([-1, 1])

                    #move weights exactly by one lsb
                    perturb_w = direction_w * lsb
                    perturb_b = direction_b * lsb

                    #evalutate f(w + delta u): forward
                    self.w += perturb_w
                    self.b += perturb_b
                    y_pred_plus = self.predict(X_sub)
                    loss_plus = self.compute_zoo_cost(y_pred_plus, y_sub, loss_func, delta=delta)

                    #evaluate f(w - delta u): backward
                    self.w -= 2 * perturb_w
                    self.b -= 2 * perturb_b
                    y_pred_minus = self.predict(X_sub)
                    loss_minus = self.compute_zoo_cost(y_pred_minus, y_sub, loss_func, delta=delta) 

                    # restore orgginal weights
                    self.w += perturb_w
                    self.b += perturb_b

                    #two points gradient estimate
                    # (2 * lsb) is always a power of 2. 
                    # In hardware, dividing by a power of 2 is NOT floating-point division; 
                    # it is just a simple "right bit-shift" (>>), making this operation virtually free!
                    grad_est_w = ((loss_plus - loss_minus) / (2 * lsb)) * direction_w
                    grad_est_b = ((loss_plus - loss_minus) / (2 * lsb)) * direction_b

                    grad_w_total += grad_est_w
                    grad_b_total += grad_est_b

                #Aggregate gradients
                # len(sub_batches) is the power of two: 32//8 = 4
                # this is also just a simple right bit-shift.
                grad_w_total /= len(sub_batches)
                grad_b_total /= len(sub_batches)

                # Update weights
                # learning rate is the power of two
                # this multiplication becomes another fast bit-shift.
                self.w -= lr * grad_w_total
                self.b -= lr * grad_b_total

                # Quantization (simulate hardware registers)
                self.w = fixed_point_quantize(self.w, total_bits, frac_bits)
                self.b = fixed_point_quantize(self.b, total_bits, frac_bits)

            #Return standard mean and sum mean that was used in the hardware
            y_pred_full = self.predict(X)
            error_full = y_pred_full - y
            if loss_func == "MSE":
                loss_hardware = np.sum(error_full**2)
                loss_epoch = np.mean(error_full**2)
            elif loss_func == "MAE":
                loss_epoch = np.mean(np.abs(error_full))
                loss_hardware = np.sum(np.abs(error_full))
            else:
                is_small = np.abs(error_full) <= delta
                squared = 0.5 * error_full**2
                linear = delta * (np.abs(error_full) - 0.5 * delta)
                loss_epoch = np.mean(np.where(is_small, squared, linear))
                loss_hardware = np.sum(np.where(is_small, squared, linear))
                
            loss_history.append(loss_epoch)
            loss_history_hardware.append(loss_hardware)

        return loss_history, loss_history_hardware
    




    def _stair_quantize(self, x, total_bits=8, frac_bits=4, alpha=1.0, beta=1.0):
        lsb = 2 ** (-frac_bits)
        
        # Find the physical hardware center (e.g., 0.0625, 0.1250)
        centers = np.round(x / lsb) * lsb
        diff = x - centers
        abs_diff = np.abs(diff)

        # Widths: a + b = lsb
        a = (alpha * lsb) / (alpha + 1.0)
        b = lsb / (alpha + 1.0)
        
        # Heights: c + d = lsb
        c = lsb / (1.0 + beta * alpha)
        d = lsb - c

        #Calculate the true mathematical slopes
        slope_trap = c / max(a, 1e-9)
        slope_transition = d / max(b, 1e-9)

        # 3. Define the zones
        # The trap extends a/2 in physical distance
        is_trap = abs_diff <= (a / 2.0)

        x_soft = np.where(
            is_trap,
            centers + diff * slope_trap, 
            centers + np.sign(diff) * ((a / 2.0) * slope_trap + (abs_diff - (a / 2.0)) * slope_transition)
        )

        # We must multiply the integer limits by LSB to get physical bounds
        qmax_physical = ((2 ** (total_bits - 1)) - 1) * lsb
        qmin_physical = -(2 ** (total_bits - 1)) * lsb
        
        x_q = np.clip(x_soft, qmin_physical, qmax_physical)

        # 5. Backward Pass (The Clamped Gradients)
        safe_slope_transition = np.clip(slope_transition, a_min=0.0, a_max=1.0)
        safe_slope_trap = np.clip(slope_trap, a_min=0.1, a_max=1.0)

        grad = np.where(
            is_trap,
            safe_slope_trap,       
            safe_slope_transition  
        )

        return x_q, grad
    

    def fit_qat_non_linear(self, X, y, epochs=500, lr=0.01, total_bits=8, frac_bits=4):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        loss_history = []
        
        for epoch in range(epochs):
            linear_progress = epoch / max(1, (epochs - 1))
            
            alpha = 0.1 + (99.9 * linear_progress)
            
            beta_straight = 1.0 / (alpha ** 2)
            beta_cliff = 10.0 
            beta = beta_straight + linear_progress * (beta_cliff - beta_straight)
            
            w_q, dw_q = self._stair_quantize(self.w, total_bits, frac_bits, alpha, beta)
            b_q, db_q = self._stair_quantize(self.b, total_bits, frac_bits, alpha, beta)


            y_pred = X @ w_q + b_q
            
            #Calculate Loss (MSE for standard Backprop)
            error = y_pred - y
            loss_history.append(np.mean(error**2))
            
            # Standard Backpropagation (Gradients of the Loss)
            # These are the gradients with respect to the QUANTIZED weights
            dL_dw_q = (2 / n_samples) * (X.T @ error) 
            dL_db_q = (2 / n_samples) * np.sum(error)
            
            # We multiply the loss gradient by the local slope of our custom staircase
            dL_dw_true = dL_dw_q * dw_q
            dL_db_true = dL_db_q * db_q

            current_lr = lr * 0.5 * (1.0 + math.cos(math.pi * (epoch / epochs)))
            
            # Update the TRUE continuous weights using the COOLING learning rate 
            self.w -= current_lr * dL_dw_true
            self.b -= current_lr * dL_db_true
            
        return loss_history
