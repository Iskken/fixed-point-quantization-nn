import numpy as np
from src.data.conditioned_dataset import generate_conditioned_regression_dataset
from src.models.linear_regression import LinearRegression
from src.quantization.quantize import fixed_point_quantize
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

condition_numbers = [1, 10, 100, 1000]
egv = [[cn, 1, 1] for cn in condition_numbers]
seeds = range(10)

results = []

for e in egv:
    print(f"Generating dataset with eigenvalues: {e}")
    for seed in seeds:
        print(f"Random seed: {seed}")
        #Generate the conditioned dataset with the specified eigenvalues and random seed
        X,y, w_true, Sigma = generate_conditioned_regression_dataset(
            n_samples=10000,
            eigenvalues=e,
            w_true=np.array([1.5432, 0.5725, 0.53125]),
            noise_std=0.01,
            random_seed=seed
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
        mse_q = np.mean((y_q - y_test)**2)

        results.append({
            'cond': e[0],
            'baseline_mse': baseline_mse,
            'quantized_mse': mse_q,
            'seed': seed,
            'mean_features':np.mean(X, axis=0),
            'std_features':np.std(X, axis=0)
        })

        print(f"Eigenvalues: {e}, Baseline MSE: {baseline_mse:.6f}, Quantized MSE: {mse_q:.6f}")
        print(f"Trained weights: {model.w}, Quantized weights: {w_q}\n")

#Plotting the results
mean_baselines = []
std_baselines = []
mean_quantized = []
std_quantized = []

for n in condition_numbers:
    subset = [r for r in results if r['cond'] == n]

    baseline_vals = [r['baseline_mse'] for r in subset]
    quantized_vals = [r['quantized_mse'] for r in subset]

    mean_baselines.append(np.mean(baseline_vals))
    std_baselines.append(np.std(baseline_vals))

    mean_quantized.append(np.mean(quantized_vals))
    std_quantized.append(np.std(quantized_vals))

plt.figure(figsize=(10,6))

plt.errorbar(
    condition_numbers,
    mean_baselines,
    yerr=std_baselines,
    marker='o',
    label='Baseline MSE'
)

plt.errorbar(
    condition_numbers,
    mean_quantized,
    yerr=std_quantized,
    marker='o',
    label='Quantized MSE'
)

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Condition Number")
plt.ylabel("Mean Squared Error")
plt.title("MSE vs Condition Number (Mean ± Std over seeds)")

plt.legend()
plt.grid(True, which="both", ls="--")
plt.show()