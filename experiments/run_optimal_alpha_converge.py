import os
from experiments.run_mlp_non_linear import plot_qat_loss_comparison
from experiments.run_mlp_non_linear import plot_schedule_behavior
from experiments.run_mlp_non_linear import plot_fp_qat_pqt_loss_comparison
from src.data.dataset import generate_complex_dataset
from experiments.run_optimal_alpha import initialize_models
from experiments.run_optimal_alpha import create_model_from_initialization
from experiments.run_optimal_alpha import train_fp_pqt
from experiments.run_optimal_alpha import plot_test_mse_comparison
from experiments.run_optimal_alpha import plot_gradient_statistics
from experiments.run_optimal_alpha import plot_gradient_vs_alpha
from src.models.mlp import MLP

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np



def train_qat_stepwise(
    qat_model,
    X_train,
    X_test,
    y_train,
    y_test,
    alpha_steps,
    lr=0.01,
    total_bits=8,
    frac_bits=4,
    beta=1.0,
    tolerance=1e-6,
    patience=100,
    max_epochs_per_alpha=10000,
    verbose=True
):
    """
    Train a QAT model using a stepwise alpha schedule.

    At each alpha:
        1. Train until convergence.
        2. Keep the learned weights.
        3. Move to the next alpha.
        4. Continue training from the previous alpha's weights.

    Beta is fixed throughout training.
    """

    print("\n" + "=" * 60)
    print("Training QAT model - Stepwise Alpha Schedule")
    print("=" * 60)

    print(f"Alpha steps: {alpha_steps}")
    print(f"Beta: {beta}")
    print(f"Learning rate: {lr}")

    # -----------------------------------------
    # 1. Train using stepwise alpha schedule
    # -----------------------------------------

    (
        loss_history_qat,
        alpha_history,
        beta_history,
        gradient_history,
        convergence_info
    ) = qat_model.fit_qat_stepwise(
        X_train,
        y_train,
        alpha_steps=alpha_steps,
        lr=lr,
        total_bits=total_bits,
        frac_bits=frac_bits,
        beta=beta,
        tolerance=tolerance,
        patience=patience,
        max_epochs_per_alpha=max_epochs_per_alpha,
        verbose=verbose
    )

    # -----------------------------------------
    # 2. Final exact quantized evaluation
    # -----------------------------------------

    y_train_qat = qat_model.predict_quantized(
        X_train,
        total_bits=total_bits,
        fractional_bits=frac_bits
    )

    y_test_qat = qat_model.predict_quantized(
        X_test,
        total_bits=total_bits,
        fractional_bits=frac_bits
    )

    train_mse_qat = np.mean(
        (y_train_qat - y_train) ** 2
    )

    test_mse_qat = np.mean(
        (y_test_qat - y_test) ** 2
    )

    # -----------------------------------------
    # 3. Return results
    # -----------------------------------------

    return {
        "schedule_type": "stepwise",

        "alpha_steps": alpha_steps,
        "beta": beta,

        "train_mse_qat": train_mse_qat,
        "test_mse_qat": test_mse_qat,

        "loss_history_qat": loss_history_qat,

        "alpha_history": alpha_history,
        "beta_history": beta_history,

        "gradient_history": gradient_history,

        "convergence_info": convergence_info,

        "model_qat": qat_model
    }





def main():

    # =========================================
    # Configuration
    # =========================================

    RESULTS_DIR = "results/complex_dataset_stepwise_alpha"

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    layer_sizes = [4, 32, 32, 16, 1]

    lr = 0.01

    total_bits = 8
    frac_bits = 4

    # Stepwise alpha schedule
    alpha_steps = [
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0
    ]

    # Beta is fixed
    beta = 1.0

    # Convergence parameters
    tolerance = 1e-6
    patience = 100
    max_epochs_per_alpha = 15000

    # =========================================
    # Generate dataset
    # =========================================

    X, y = generate_complex_dataset(
        n_features=4,
        n_samples=2000,
        freq_list=(3.0, 6.0),
        noise_std=0.01,
        random_seed=42
    )

    # =========================================
    # Train/test split
    # =========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # =========================================
    # Initialize models
    # =========================================

    initial_weights, initial_biases = initialize_models(
        layer_sizes
    )

    # =========================================
    # Floating-point / PQT baseline
    # =========================================

    fp_model = create_model_from_initialization(
        layer_sizes,
        initial_weights,
        initial_biases
    )

    fp_pqt_result = train_fp_pqt(
        fp_model,
        X_train,
        X_test,
        y_train,
        y_test,
        epochs=49172,
        lr=lr,
        total_bits=total_bits,
        frac_bits=frac_bits,
        verbose=True
    )

    # =========================================
    # Stepwise QAT
    # =========================================

    qat_model = create_model_from_initialization(
        layer_sizes,
        initial_weights,
        initial_biases
    )

    result = train_qat_stepwise(
        qat_model,
        X_train,
        X_test,
        y_train,
        y_test,
        alpha_steps=alpha_steps,
        lr=lr,
        total_bits=total_bits,
        frac_bits=frac_bits,
        beta=beta,
        tolerance=tolerance,
        patience=patience,
        max_epochs_per_alpha=max_epochs_per_alpha,
        verbose=True
    )

    # Put into list because your existing plotting
    # functions expect a list of QAT results.
    qat_results = [result]

    # =========================================
    # Plot results
    # =========================================

    for result in qat_results:

        plot_schedule_behavior(
            result,
            RESULTS_DIR
        )

        plot_gradient_statistics(
            result,
            RESULTS_DIR
        )

        plot_gradient_vs_alpha(
            result,
            RESULTS_DIR
        )

    plot_qat_loss_comparison(
        qat_results,
        RESULTS_DIR
    )

    plot_test_mse_comparison(
        fp_pqt_result,
        qat_results,
        RESULTS_DIR
    )

    for i in range(len(qat_results)):

        plot_fp_qat_pqt_loss_comparison(
            fp_pqt_result,
            qat_results[i],
            RESULTS_DIR
        )

    # =========================================
    # Print convergence information
    # =========================================

    print("\n" + "=" * 60)
    print("STEPWISE CONVERGENCE RESULTS")
    print("=" * 60)

    for info in result["convergence_info"]:

        print(
            f"Alpha = {info['alpha']:>4} | "
            f"Epochs = {info['epochs']:>5} | "
            f"Final loss = {info['final_loss']:.8f}"
        )

    total_epochs = sum(
        info["epochs"]
        for info in result["convergence_info"]
    )

    print("-" * 60)
    print(f"Total training epochs: {total_epochs}")

    print ("Final floating-point train MSE: "
           f"{fp_pqt_result['train_mse_fp']:.6f}")

    print(
        f"Final floating-point test MSE: "
        f"{fp_pqt_result['test_mse_fp']:.6f}"
    )
    print(
        f"Final quantized train MSE: "
        f"{result['train_mse_qat']:.6f}"
    )

    print(
        f"Final pqt train MSE: "
        f"{fp_pqt_result['train_mse_pqt']:.6f}"
    )

    print(
        f"Final pqt test MSE: "
        f"{fp_pqt_result['test_mse_pqt']:.6f}"
    )

    print(
        f"Final quantized test MSE: "
        f"{result['test_mse_qat']:.6f}"
    )

    print("=" * 60)

    print("\nExperiments completed.")
    print(
        f"Results saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()