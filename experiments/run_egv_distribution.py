import numpy as np
from src.data.conditioned_dataset import generate_conditioned_regression_dataset
from src.models.linear_regression import LinearRegression
from src.quantization.quantize import fixed_point_quantize
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#This experiment is designed to investigate the impact of the distribution of eigenvalues 
#on the sensitivity to quantization in linear regression models.

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

condition_numbers = [2, 10, 100, 1000]
seeds = range(10)

results = []

set_sizes = [1, 2, 6, 14, 30]


for s in set_sizes:
    for c in condition_numbers:
        egv_set = generate_egv_sets(max_egv=c, min_egv=1, num_sets=5, set_size=s)
        print(f"Set size: {s}, Eigenvalue sets: {egv_set}")
        
        for e in egv_set:
            #Generate the conditioned dataset with the specified eigenvalues and random seed
            X,y, w_true, Sigma = generate_conditioned_regression_dataset(
                n_samples=1000,
                eigenvalues=e,
                w_true=np.ones(len(e)) * 1.5,
                noise_std=0.1,
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
            w_q = fixed_point_quantize(model.w, total_bits=8, fractional_bits=4)
            y_q = X_test @ w_q + model.b
            mse_q = np.mean(((y_q - y_test)**2)/len(egv_set))

            results.append({
                'cond': e[0],
                'set_size': s,
                'inter_eigenvalues': e[1:-1],
                'baseline_mse': baseline_mse,
                'quantized_mse': mse_q,
                'mean_features':np.mean(X, axis=0),
                'std_features':np.std(X, axis=0)
            })

            print(f"Eigenvalues: {e}, Baseline MSE: {baseline_mse:.6f}, Quantized MSE: {mse_q:.6f}")

#Plotting the results
mean_ratio = {}
std_ratio = {}

for s in set_sizes:
    for c in condition_numbers:
        subset = [r for r in results if r['set_size'] == s and r['cond'] == c]

        if len(subset) == 0:
            continue

        ratios = [
            r['quantized_mse'] / r['baseline_mse']
            for r in subset
        ]

        mean_ratio[(s,c)] = np.mean(ratios)
        std_ratio[(s,c)] = np.std(ratios)
        
plt.figure(figsize=(10,6))

for s in set_sizes:
    x = []
    y = []
    yerr = []

    for c in condition_numbers:
        if (s,c) in mean_ratio:
            x.append(c)
            y.append(mean_ratio[(s,c)])
            yerr.append(std_ratio[(s,c)])

    plt.errorbar(
        x, y,
        yerr=yerr,
        marker='o',
        label=f'dim={s+2}'
    )

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Condition Number")
plt.ylabel("Quantization Error Ratio (MSE_q / MSE_base)")
plt.title("Quantization Sensitivity vs Condition Number")

plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()