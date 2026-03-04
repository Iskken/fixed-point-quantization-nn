import numpy as np
import matplotlib.pyplot as plt
from src.data.dataset import generate_regression_dataset
from src.quantization.quantize import fixed_point_quantize


#In order to test this, run the script from the project root
#python -m "experiments.test_quantize"



w_true_val = [1.54321] 
b_true_val = 0.0

X, y, w_true, b_true = generate_regression_dataset(
    w_true=w_true_val,
    n_samples=200,
    noise_std=0.2 
)

# Experiment with 4-bit total (very low) and 2 fractional bits
TOTAL_BITS = 4
FRAC_BITS = 2

w_quantized = fixed_point_quantize(w_true, total_bits=TOTAL_BITS, fractional_bits=FRAC_BITS)

print(f"Original Weight:  {w_true[0]}")
print(f"Quantized Weight: {w_quantized[0]} (using Q{TOTAL_BITS-FRAC_BITS-1}.{FRAC_BITS} format)")
print(f"Difference (Noise): {abs(w_true[0] - w_quantized[0]):.4f}")




# 3. Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X[:,0], y, alpha=0.3, label="Data Samples")

# Generate line coordinates
x_line = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_true_line = w_true[0] * x_line + b_true
y_quant_line = w_quantized[0] * x_line + b_true

# Plot comparison
plt.plot(x_line, y_true_line, color="red", linewidth=2, label=f"True Line (FP32: {w_true[0]})")
plt.plot(x_line, y_quant_line, color="green", linestyle="--", linewidth=2, 
         label=f"Quantized Line ({TOTAL_BITS}-bit: {w_quantized[0]})")

plt.title(f"Impact of {TOTAL_BITS}-bit Quantization on Model Slope")
plt.xlabel("Input X")
plt.ylabel("Output y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()