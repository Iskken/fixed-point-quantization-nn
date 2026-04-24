from src.data.dataset import generate_regression_dataset
from src.data.conditioned_dataset import generate_conditioned_regression_dataset
from src.models.linear_regression import LinearRegression
from src.quantization.quantize import fixed_point_quantize
import matplotlib.pyplot as plt
import numpy as np

#How to run this experiment:
# python -m experiments.run_gradient_deadlock_solution



# this pipeline will run two experiments: one with gradient scaling and other is error accumulation. We will compare the convergence of both methods and see if they can solve the deadlock issue in quantized training.


def plot_convergence(loss_std = None, 
                     loss_qat = None, 
                     loss_ptq_final = None, 
                     loss_qat_accumulation = None, 
                     loss_zoo = None, 
                     loss_zoo_hardware = None, 
                     loss_func="MSE", 
                     scale = 1.02): 
    
    plt.figure(figsize=(10, 6))

    # Plot Standard and QAT history
    if loss_std is not None:
        plt.plot(loss_std, label='Standard (FP64)', color='blue', linewidth=2)
    
    if loss_qat is not None:
        plt.plot(loss_qat, label=f'Quantization-Aware (Gradiend scalind: {scale})', color='green', linestyle='--')

    if loss_qat_accumulation is not None:
        plt.plot(loss_qat_accumulation, label='QAT with Error Accumulation', color='orange', linestyle='-.')
    
    if loss_zoo is not None:
        plt.plot(loss_zoo, label='ZOO (standard)', color='purple', linestyle='-.')

    if loss_zoo_hardware is not None:
        plt.plot(loss_zoo_hardware, label='ZOO (hardware)', color='black', linestyle=':')

    # Plot the PTQ failure as a reference point
    if loss_ptq_final is not None:
         plt.axhline(y=loss_ptq_final, color='red', linestyle=':', label='Post-Training Quant (PTQ)')

    # Dynamic Y-Axis Labels 
    label_map = {
        "MSE": "Mean Squared Error (Log Scale)",
        "MAE": "Mean Absolute Error (Log Scale)",
        "Huber": "Huber Loss (Log Scale)"
    }
    # Safely get the label, default to the raw name if it's not in the dictionary
    y_label = label_map.get(loss_func, f"{loss_func} Loss (Log Scale)")

    plt.yscale('log') # Log scale makes the differences near the bottom visible
    
    # --- NEW: Added loss_func to the title ---
    plt.title(f'Convergence ({loss_func}): FP64 vs. QAT vs. PTQ | {TOTAL_BITS}-tot_bits, {FRAC_BITS}-frac_bits, LR={LEARNING_RATE}')
    plt.xlabel('Epochs')
    
    # --- NEW: Apply the dynamic label ---
    plt.ylabel(y_label) 
    
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()



def train_qat_with_gradient_scaling(X, y, lr, total_bits, frac_bits, scaling_factor, epochs=500, loss_func = "MSE"):
    print("\n--- Training with Gradient Scaling ---")
    model_qat = LinearRegression()
    loss_qat = model_qat.fit_normal_descent_quantize_gradient_scaling(X, y, epochs=epochs, lr=lr, total_bits=total_bits, frac_bits=frac_bits, scaling_factor=scaling_factor, loss_func=loss_func)  
    return loss_qat

def train_qat_with_error_accumulation(X, y, lr, total_bits, frac_bits, epochs=500, loss_func = "MSE"):
    print("\n--- Training with Error Accumulation ---")
    model_qat_accum = LinearRegression()
    loss_qat_accum = model_qat_accum.fit_error_accumulation(X, y, epochs=epochs, lr=lr, total_bits=total_bits, frac_bits=frac_bits, loss_func=loss_func)  
    return loss_qat_accum

def train_qat_with_zoo(X, y, lr_shift, total_bits, frac_bits, epochs=500, loss_func="MSE"):
    print("\n--- Training with Zero order optimization ---")
    model_zoo = LinearRegression()
    loss_zoo, loss_zoo_hardware = model_zoo.fit_zoo(X, y, epochs = epochs, lr_shift = lr_shift, total_bits = total_bits, frac_bits = frac_bits, loss_func = loss_func)
    return loss_zoo, loss_zoo_hardware

def train_baseline_model(X, y, lr, epochs=500, loss_func="MSE"):
    print("--- Training Baseline Model (FP64) ---")
    model_std = LinearRegression()
    loss_std = model_std.fit_gradient_descent(X, y, epochs=epochs, lr=lr, loss_func=loss_func)
    return model_std, loss_std

def train_with_ptq(model_std, X, y, total_bits, frac_bits, loss_func = "MSE", delta = 1.0):
    print("\n--- Training with Post-Training Quantization (PTQ) ---")
    w_ptq = fixed_point_quantize(model_std.w, total_bits=total_bits, fractional_bits=frac_bits)
    b_ptq = fixed_point_quantize(model_std.b, total_bits=total_bits, fractional_bits=frac_bits)
    y_pred_ptq = X @ w_ptq + b_ptq
    error = y_pred_ptq - y
    if loss_func == "MSE":
            loss = np.mean(error**2)

    elif loss_func == "MAE":
        loss = np.mean(np.abs(error))

    elif loss_func == "Huber":
        # Create a mask to separate small errors from huge outliers
        is_small_error = np.abs(error) <= delta
        
        # Calculate Loss
        squared_loss = 0.5 * error**2
        linear_loss = delta * (np.abs(error) - 0.5 * delta)
        loss = np.mean(np.where(is_small_error, squared_loss, linear_loss))
    else:
        raise ValueError(f"Unknown loss function: {loss_func}")
    return loss


if __name__ == "__main__":
    np.random.seed(42)


    TOTAL_BITS = 8
    FRAC_BITS = 4
    LEARNING_RATE = 0.0001
    EPOCHS = 500
    LOSS_FUNC = "Huber"
    GR_SCALE = 10
    lr_shift = 6  ## learning rate for the zoo will be: 0.015625    
    # creating the sample dataset


    # #well conditioned dataset
    # w_true_val = [1.54321] 
    # b_true_val = 0.0
    # X, y, w_true, b_true = generate_regression_dataset(w_true=w_true_val, b_true=b_true_val, n_samples=1000)


    #ill conditione dataset
    n_samples = 1000
    dim = 3
    condition_number = 10000
    eigenvalues = np.logspace(0, np.log10(condition_number), num=dim)
    w_true = np.random.randn(dim)
    X, y, w_true, Sigma = generate_conditioned_regression_dataset(
        n_samples=n_samples,
        eigenvalues=eigenvalues,
        w_true=w_true,
        noise_std=0.01,
        random_seed=42
    )


    # training the baseline model (gold standard)
    model_std, loss_std = train_baseline_model(X, y, LEARNING_RATE, epochs=EPOCHS, loss_func=LOSS_FUNC)

    # Train with post-training quantization (PTQ) Quantize the final result of model_std and evaluate the loss
    loss_ptq_final = train_with_ptq(model_std, X, y, TOTAL_BITS, FRAC_BITS, loss_func=LOSS_FUNC)

    # Run Qat training with gradient scaling
    loss_qat_gradient_scaling = train_qat_with_gradient_scaling(X, y, LEARNING_RATE, TOTAL_BITS, FRAC_BITS, scaling_factor=GR_SCALE, epochs=EPOCHS, loss_func=LOSS_FUNC)  # You can experiment with different scaling factors like 10.0, 20.0, etc.

    # Run Qat training with error accumulation
    loss_qat_error_accumulation = train_qat_with_error_accumulation(X, y, LEARNING_RATE, TOTAL_BITS, FRAC_BITS, epochs=EPOCHS, loss_func=LOSS_FUNC)

    # Run QAT training with zoo
    loss_zoo, loss_zoo_hardware = train_qat_with_zoo(X, y, lr_shift, TOTAL_BITS, FRAC_BITS, epochs=EPOCHS, loss_func=LOSS_FUNC)

    #Visualize the convergence of the three methods
    plot_convergence(loss_std, loss_qat_gradient_scaling, loss_ptq_final, loss_qat_error_accumulation, loss_zoo, loss_zoo_hardware, loss_func=LOSS_FUNC, scale=GR_SCALE)