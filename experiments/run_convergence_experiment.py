from src.data.dataset import generate_regression_dataset
from src.models.linear_regression import LinearRegression
from src.quantization.quantize import fixed_point_quantize
import matplotlib.pyplot as plt
import numpy as np

#How to run this experiment:
# python -m experiments.run_convergence_experiment

#Let's run the three-way convergence test
# 1 Standard Gradient Descent (FP64)
# 2 Standard Gradient Descent with post-training quantization (PTQ)
# 3 Quantization-Aware Training (QAT) with 8-bit quantization

TOTAL_BITS = 8
FRAC_BITS = 4
LEARNING_RATE = 0.1
# creating the sample dataset
w_true_val = [1.54321] 
b_true_val = 0.0
X, y, w_true, b_true = generate_regression_dataset(w_true=w_true_val, b_true=b_true_val, n_samples=1000)

# training the baseline model (gold standard)
print("--- Training Baseline Model (FP64) ---")
model_std = LinearRegression()
loss_std = model_std.fit_gradient_descent(X, y, epochs=500, lr=LEARNING_RATE)

# Train with post-training quantization (PTQ) Quantize the final result of model_std and evaluate the loss
print("\n--- Training with Post-Training Quantization (PTQ) ---")
w_ptq = fixed_point_quantize(model_std.w, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
b_ptq = fixed_point_quantize(model_std.b, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)
y_pred_ptq = X @ w_ptq + b_ptq
loss_ptq_final = np.mean((y_pred_ptq - y)**2)



# Run Qat training with 8-bit quantization
print("\n--- Training with Quantization-Aware Training (QAT) ---")
model_qat = LinearRegression()
loss_qat = model_qat.fit_normal_descent_quantize(X, y, epochs=500, lr=LEARNING_RATE, total_bits=TOTAL_BITS, frac_bits=FRAC_BITS)  



#Visualize the convergence of the three methods
# --- Plotting ---
plt.figure(figsize=(10, 6))

# Plot Standard and QAT history
plt.plot(loss_std, label='Standard (FP64)', color='blue', linewidth=2)
plt.plot(loss_qat, label='Quantization-Aware (QAT 8-bit)', color='green', linestyle='--')

# Plot the PTQ failure as a reference point
plt.axhline(y=loss_ptq_final, color='red', linestyle=':', label='Post-Training Quant (PTQ)')

plt.yscale('log') # Log scale makes the differences near the bottom visible
plt.title(f'Convergence Comparison: FP64 vs. QAT vs. PTQ:  {TOTAL_BITS}-tot_bits, {FRAC_BITS}-frac_bits, LR={LEARNING_RATE}')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()

print(f"Final Loss FP64: {loss_std[-1]:.6f}")
print(f"Final Loss QAT:  {loss_qat[-1]:.6f}")
print(f"Final Loss PTQ:  {loss_ptq_final:.6f}")