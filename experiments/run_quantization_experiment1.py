from src.data.dataset import generate_regression_dataset
from src.quantization.quantize import fixed_point_quantize
from src.models.linear_regression import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# creating the sample dataset
w_true_val = [1.54321] 
b_true_val = 0.0

X, y, w_true, b_true = generate_regression_dataset(
    w_true=w_true_val,
    n_samples=200,
    noise_std=0.2 
)


# training the baseline model (gold standard)
print("--- Training Baseline Model (FP64) ---")
model = LinearRegression()
model.fit_gradient_descent(X, y, epochs=500, lr=0.01)

#saving the perfect weight
w_fp = model.w.copy()
b_fp = model.b


#calculate the baseline loss(MSE)
y_pred_fp = model.predict(X); 
baseline_loss = np.mean((y_pred_fp-y)**2)
print(f"Baseline Loss: {baseline_loss:.6f}\n")



#calculate with quantization
total_bits = 8
results = []


print(f"--- Sweeping Fractional Bits (Total Bits = {total_bits}) ---")
print(f"{'Frac Bits':<10} | {'Loss':<12} | {'Status'}")
print("-" * 40)

for f_bits in range(total_bits + 1):
    # Quantize
    w_q = fixed_point_quantize(w_fp, total_bits=total_bits, fractional_bits=f_bits)
    b_q = fixed_point_quantize(b_fp, total_bits=total_bits, fractional_bits=f_bits)
    
    # Calculate the hardware limits in FLOAT terms
    scaling_factor = 2**f_bits
    max_float = (2**(total_bits - 1) - 1) / scaling_factor
    min_float = -(2**(total_bits - 1)) / scaling_factor
    
    # Inject and Predict
    model.w = w_q
    model.b = b_q
    y_pred_q = model.predict(X)
    q_loss = np.mean((y_pred_q - y)**2)
    
  
    status = "Normal"
    
    # Check for Clipping: Did we hit the float ceiling/floor?
    is_clipped = np.any(w_q >= max_float) or np.any(w_q <= min_float) or \
                 (b_q >= max_float) or (b_q <= min_float)

    if is_clipped and q_loss > baseline_loss * 1.5:
          status = "Clipped (Overflow/Range Error)"
    elif q_loss > baseline_loss * 1.5:
        status = "High Noise (Precision Error)"
    elif np.all(w_q == 0) and np.all(b_q == 0):
        status = "Underflow (Weights rounded to zero)"
        
    print(f"{f_bits:<10} | {q_loss:<12.6f} | {status}")
    results.append((f_bits, q_loss))

# 4. Visualization
f_bits_val, loss_val = zip(*results)
plt.figure(figsize=(10, 6))
plt.plot(f_bits_val, loss_val, marker='o', linestyle='-', color='b')
plt.axhline(y=baseline_loss, color='r', linestyle='--', label='Baseline (FP64)')
plt.yscale('log') # Use log scale to see the big jumps in error
plt.title(f"Quantization Sensitivity (Total Bits: {total_bits})")
plt.xlabel("Fractional Bits")
plt.ylabel("Mean Squared Error (Log Scale)")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()






