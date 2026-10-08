import os
from experiments.run_mlp_non_linear import plot_qat_loss_comparison
from experiments.run_mlp_non_linear import plot_schedule_behavior
from experiments.run_mlp_non_linear import plot_fp_qat_pqt_loss_comparison
from src.data.dataset import generate_complex_dataset
from src.models.mlp import MLP

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np



def initialize_models(layer_sizes):
    """
    Initialize models and preserve the common
    initial weights for reproducible comparisons.
    """

    base_model = MLP(layer_sizes)

    initial_weights = [
        w.copy() for w in base_model.weights
    ]

    initial_biases = [
        b.copy() for b in base_model.biases
    ]

    return initial_weights, initial_biases

def create_model_from_initialization(
    layer_sizes,
    initial_weights,
    initial_biases):

    model = MLP(layer_sizes)

    model.weights = [
        w.copy() for w in initial_weights
    ]

    model.biases = [
        b.copy() for b in initial_biases
    ]

    return model


def train_qat(
    qat_model,
    schedule_type,
    X_train,
    X_test,
    y_train,
    y_test,
    epochs=20000,
    lr=0.01,
    total_bits=8,
    frac_bits=4,
    alpha_start=0.1,
    alpha_end=100.0,
    transition_epochs=20000,
    verbose=True,
    learning_rate_schedule=None,
    snapshot_alphas=None
):
    """
    Train the QAT model using an alpha-only schedule
    with beta fixed to 1.0.

    Gradient statistics and distributions are also
    collected during training.
    """

    print("\n" + "=" * 60)
    print(f"Training QAT model - {schedule_type}")
    print("=" * 60)

    # -----------------------------------------
    # 1. Train QAT model
    # -----------------------------------------

    (
        loss_history_qat,
        alpha_history,
        beta_history,
        gradient_history,
        gradient_snapshots
    ) = qat_model.fit_qat_non_linear(
        X_train,
        y_train,
        epochs=epochs,
        lr=lr,
        total_bits=total_bits,
        frac_bits=frac_bits,
        schedule_type=schedule_type,
        alpha_start=alpha_start,
        alpha_end=alpha_end,
        transition_epochs = transition_epochs,
        verbose=verbose,
        learning_rate_schedule=learning_rate_schedule,
        snapshot_alphas=snapshot_alphas
    )

    # -----------------------------------------
    # 2. Final quantized evaluation
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
        "schedule_type": schedule_type,

        "train_mse_qat": train_mse_qat,
        "test_mse_qat": test_mse_qat,

        "loss_history_qat": loss_history_qat,

        "alpha_history": alpha_history,
        "beta_history": beta_history,

        "gradient_history": gradient_history,
        "gradient_snapshots": gradient_snapshots,

        "model_qat": qat_model
    }




def train_fp_pqt(
    fp_model,
    X_train,
    X_test,
    y_train,
    y_test,
    epochs=20000,
    lr=0.01,
    total_bits=8,
    frac_bits=4,
    verbose=True
):
    """
    Train the floating-point model and evaluate:

    1. Floating-point inference
    2. Post-quantization (PQT)
    """

    # -----------------------------------------
    # 1. Train floating-point model
    # -----------------------------------------

    print("\n" + "=" * 60)
    print("Training floating-point model")
    print("=" * 60)

    fp_model.fit(
        X_train,
        y_train,
        epochs=epochs,
        lr=lr,
        verbose=verbose
    )

    # -----------------------------------------
    # 2. Floating-point predictions
    # -----------------------------------------

    y_train_fp = fp_model.predict(X_train)
    y_test_fp = fp_model.predict(X_test)

    train_mse_fp = np.mean(
        (y_train_fp - y_train) ** 2
    )

    test_mse_fp = np.mean(
        (y_test_fp - y_test) ** 2
    )

    # -----------------------------------------
    # 3. PQT predictions
    # -----------------------------------------

    y_train_pqt = fp_model.predict_quantized(
        X_train,
        total_bits=total_bits,
        fractional_bits=frac_bits
    )

    y_test_pqt = fp_model.predict_quantized(
        X_test,
        total_bits=total_bits,
        fractional_bits=frac_bits
    )

    train_mse_pqt = np.mean(
        (y_train_pqt - y_train) ** 2
    )

    test_mse_pqt = np.mean(
        (y_test_pqt - y_test) ** 2
    )

    # -----------------------------------------
    # 4. Return results
    # -----------------------------------------

    return {
        "train_mse_fp": train_mse_fp,
        "test_mse_fp": test_mse_fp,

        "train_mse_pqt": train_mse_pqt,
        "test_mse_pqt": test_mse_pqt,

        "loss_history_fp": fp_model.loss_history,

        "model_fp": fp_model
    }



def plot_test_mse_comparison(fp_pqt_result, qat_results, results_dir):
    """
    Compare QAT test MSE across all scheduling methods,
    with FP and PQT shown as global reference lines.
    """

    schedules = [
        qat_result["schedule_type"]
        for qat_result in qat_results
    ]

    # Global baselines
    fp_mse = fp_pqt_result["test_mse_fp"]
    pqt_mse = fp_pqt_result["test_mse_pqt"]

    # QAT result for each schedule
    qat_mse = [
        qat_result["test_mse_qat"]
        for qat_result in qat_results
    ]

    x = np.arange(3)

    bar_list = [qat_mse[0], fp_mse, pqt_mse]

    plt.figure(figsize=(12, 6))


    bars = plt.bar(
            x,
            bar_list,
            width=0.5,
            label="QAT"
        )

    plt.bar_label(
            bars,
            fmt="%.4f",
            padding=3
        )
    # -----------------------------------------
    # Labels
    # -----------------------------------------

    plt.xticks(
        x,
        ["QAT", "FP", "PQT"],
        rotation=20
    )

    plt.xlabel("Training Type")
    plt.ylabel("Test MSE")

    plt.title(
        "Test MSE Comparison of Training Types"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "test_mse_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def plot_gradient_statistics(
        result,
        results_dir
    ):
    """
    Plot mean, median, and maximum absolute gradient
    throughout QAT training.
    """

    gradient_history = result["gradient_history"]

    epochs = [
        item["epoch"]
        for item in gradient_history
    ]

    mean_grad = [
        item["mean_abs_gradient"]
        for item in gradient_history
    ]

    median_grad = [
        item["median_abs_gradient"]
        for item in gradient_history
    ]

    max_grad = [
        item["max_abs_gradient"]
        for item in gradient_history
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        mean_grad,
        label="Mean |gradient|"
    )

    plt.plot(
        epochs,
        median_grad,
        label="Median |gradient|"
    )

    plt.plot(
        epochs,
        max_grad,
        label="Max |gradient|"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Absolute gradient")
    plt.title(
        f"Gradient Statistics - "
        f"{result['schedule_type']}"
    )

    plt.yscale("log")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(
        results_dir,
        f"gradient_statistics_"
        f"{result['schedule_type']}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

def plot_gradient_vs_alpha(
    result,
    results_dir
):
    """
    Plot absolute gradient statistics as a function
    of alpha.
    """

    gradient_history = result["gradient_history"]

    alpha = [
        item["alpha"]
        for item in gradient_history
    ]

    mean_grad = [
        item["mean_abs_gradient"]
        for item in gradient_history
    ]

    median_grad = [
        item["median_abs_gradient"]
        for item in gradient_history
    ]

    max_grad = [
        item["max_abs_gradient"]
        for item in gradient_history
    ]

    plt.figure(figsize=(10, 6))

    plt.plot(
        alpha,
        mean_grad,
        label="Mean |gradient|"
    )

    plt.plot(
        alpha,
        median_grad,
        label="Median |gradient|"
    )

    plt.plot(
        alpha,
        max_grad,
        label="Max |gradient|"
    )

    plt.xlabel("Alpha")
    plt.ylabel("Absolute gradient")
    plt.title(
        f"Gradient Magnitude vs Alpha - "
        f"{result['schedule_type']}"
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = os.path.join(
        results_dir,
        f"gradient_vs_alpha_"
        f"{result['schedule_type']}.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()


def plot_gradient_distributions_combined(
    result,
    results_dir,
    bins=100
):
    """
    Plot absolute gradient distributions for all
    captured alpha values in one figure.
    """

    gradient_snapshots = result["gradient_snapshots"]

    n_snapshots = len(gradient_snapshots)

    if n_snapshots == 0:
        return

    fig, axes = plt.subplots(
        n_snapshots,
        1,
        figsize=(10, 4 * n_snapshots)
    )

    if n_snapshots == 1:
        axes = [axes]

    for ax, (target_alpha, snapshot) in zip(
        axes,
        gradient_snapshots.items()
    ):

        gradients = snapshot["gradients"]

        ax.hist(
            gradients,
            bins=bins,
            density=True
        )

        ax.set_xlabel("|Gradient|")
        ax.set_ylabel("Density")

        ax.set_title(
            f"α = {target_alpha:.2f} "
            f"(actual α = "
            f"{snapshot['alpha']:.4f}, "
            f"epoch = {snapshot['epoch']})"
        )

        ax.set_yscale("log")

        ax.grid(
            True,
            alpha=0.3
        )

    fig.suptitle(
        f"Absolute Gradient Distributions - "
        f"{result['schedule_type']}",
        fontsize=14
    )

    plt.tight_layout()

    filename = os.path.join(
        results_dir,
        f"gradient_distributions_combined_"
        f"{result['schedule_type']}.png"
    )

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def main():

    # =========================================
    # Configuration
    # =========================================

    RESULTS_DIR = "results/complex_dataset_optimal_alpha_and_lr"

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    layer_sizes = [4, 32, 32, 16, 1]

    epochs = 50000
    lr = 0.01

    total_bits = 8
    frac_bits = 4

    alpha_start = 0.1
    alpha_end = 10.0
    beta_start = 0.1
    beta_end = 100.0

    transition_epochs = 10000

    # if the schedulers will start with prefix "alpha_", then the scheduler will be applied to alpha, and beta will remain constant at 1.0.
    schedules = [
        "alpha_linear"
        # "alpha_step",
        # "alpha_warm_up"
    ]

    learning_rate_schedule = "alpha_dependent"

    snapshot_alphas = [0.1, 0.5, 1, 2, 5, 10]

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
    # Initialize models with the same initial weights and biases
    # =========================================
    initial_weights, initial_biases = initialize_models(layer_sizes)

    # FP
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
        epochs=epochs,
        lr=lr,
        total_bits=total_bits,
        frac_bits=frac_bits,
        verbose=True
    )


    qat_results = []
    

    for schedule_type in schedules:

        result = train_qat(
            create_model_from_initialization(
                layer_sizes,
                initial_weights,
                initial_biases
            ),
            schedule_type,
            X_train,
            X_test,
            y_train,
            y_test,
            epochs=epochs,
            lr=lr,
            total_bits=total_bits,
            frac_bits=frac_bits,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            transition_epochs=transition_epochs,
            verbose=True,
            learning_rate_schedule=learning_rate_schedule,
            snapshot_alphas=snapshot_alphas
        )

        qat_results.append(result)

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

        plot_gradient_distributions_combined(
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
    # Print final table
    # =========================================

    # print_results_table([result])

    print("\nExperiments completed.")
    print(
        f"Results saved to: {RESULTS_DIR}"
    )


if __name__ == "__main__":
    main()
