import numpy as np
from sklearn.model_selection import train_test_split
from src.quantization.quantize import fixed_point_quantize
from src.models.linear_regression import LinearRegression
from src.data.conditioned_dataset import generate_conditioned_regression_dataset
import matplotlib.pyplot as plt

#This experiment is designed to investigate the impact of the condition number
#on the sensitivity to quantization in linear regression models.
#We will generate datasets with identical condition numbers, 
#but lambda_max and lambda_min will be scaled by different factors to see 
#if the absolute scale of the eigenvalues has any impact on the MSE after quantization.


max_egv = 10
min_egv = 2
scaling_factors = [0.01, 0.1, 1, 10, 100]
seeds = range(10)

results = []

def generate_egv_sets(max_egv, min_egv, num_sets, set_size):
    egv_sets = []
    for _ in range(num_sets):
        egv = np.concatenate([
            [max_egv],
            np.random.uniform(min_egv, max_egv, set_size),
            [min_egv]
        ])
        egv_sets.append(egv)
    return egv_sets

for f in scaling_factors:
    egv_set = generate_egv_sets(max_egv=max_egv * f, min_egv=min_egv*f, num_sets=5, set_size=10)
    for e in egv_set:
        X,y, w_true, Sigma = generate_conditioned_regression_dataset(
                n_samples=1000,
                eigenvalues=e,
                w_true=np.random.uniform(-2, 2, size=len(e)),
                noise_std=0.01,
                random_seed=42
            )
        
        #Split the dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        #Train the linear regression model using gradient descent and evaluate the baseline MSE
        model = LinearRegression()
        model.fit_gradient_descent(X_train, y_train, epochs=20000, lr=0.0005)
        y_pred = model.predict(X_test)
        baseline_mse = np.mean((y_pred - y_test)**2)

        #Quantize the trained weights and evaluate the MSE with quantized weights
        w_q = fixed_point_quantize(model.w, total_bits=32, fractional_bits=16)
        y_q = X_test @ w_q + model.b
        mse_q = np.mean((y_q - y_test)**2)

        results.append({
            'cond': e[0],
            'scale': f,
            'inter_eigenvalues': e[1:-1],
            'baseline_mse': baseline_mse,
            'quantized_mse': mse_q,
            'mean_features':np.mean(X, axis=0),
            'std_features':np.std(X, axis=0)
        })

        print(f"Eigenvalues: {e}, Baseline MSE: {baseline_mse:.6f}, Quantized MSE: {mse_q:.6f}")
    
mean_baselines = []
quant_baselines = []

for s in scaling_factors:
    subset = [r for r in results if r['scale'] == s]

    mean_b = np.mean([r['baseline_mse'] for r in subset])
    quant_b = np.mean([r['quantized_mse'] for r in subset])

    mean_baselines.append(mean_b)
    quant_baselines.append(quant_b)

ratio = np.array(quant_baselines) / np.array(mean_baselines)

plt.figure(figsize=(10, 6))
plt.plot(scaling_factors, ratio, marker='o')

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Scaling Factor")
plt.ylabel("Quantization Sensitivity (MSE_q / MSE_base)")
plt.title("Scale vs Quantization Sensitivity")

plt.grid(True, which="both", ls="--")
plt.show()