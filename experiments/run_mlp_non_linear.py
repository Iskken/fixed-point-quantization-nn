import os

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
    verbose=True
):
    """
    Train the QAT model using the specified alpha/beta
    scheduling strategy and evaluate the final model
    using actual fixed-point quantization.
    """

    # -----------------------------------------
    # 1. Train QAT model
    # -----------------------------------------

    print("\n" + "=" * 60)
    print(f"Training QAT model - {schedule_type}")
    print("=" * 60)

    (
        loss_history_qat,
        alpha_history,
        beta_history
    ) = qat_model.fit_qat_non_linear(
        X_train,
        y_train,
        epochs=epochs,
        lr=lr,
        total_bits=total_bits,
        frac_bits=frac_bits,
        schedule_type=schedule_type,
        verbose=verbose
    )

    # -----------------------------------------
    # 2. Final QAT evaluation
    # -----------------------------------------

    # The model still contains floating-point
    # master weights.
    #
    # predict_quantized() converts them to the
    # actual fixed-point representation during
    # inference.

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

def plot_schedule_behavior(result, results_dir):
    """
    Plot alpha, beta and QAT training loss for one schedule.
    """

    schedule_type = result["schedule_type"]

    alpha_history = result["alpha_history"]
    beta_history = result["beta_history"]
    loss_history = result["loss_history_qat"]

    schedule_dir = os.path.join(
        results_dir,
        schedule_type
    )

    os.makedirs(schedule_dir, exist_ok=True)

    epochs = np.arange(len(loss_history))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # -----------------------------------------
    # Alpha
    # -----------------------------------------

    axes[0].plot(epochs, alpha_history)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Alpha")
    axes[0].set_title(
        f"Alpha Schedule - {schedule_type}"
    )
    axes[0].grid(True)

    # -----------------------------------------
    # Beta
    # -----------------------------------------

    axes[1].plot(epochs, beta_history)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Beta")
    axes[1].set_title(
        f"Beta Schedule - {schedule_type}"
    )
    axes[1].grid(True)

    # -----------------------------------------
    # QAT training loss
    # -----------------------------------------

    axes[2].plot(epochs, loss_history)

    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE")
    axes[2].set_title(
        f"QAT Training Loss - {schedule_type}"
    )
    axes[2].grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            schedule_dir,
            "schedule_behavior.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()



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

    x = np.arange(len(schedules))

    plt.figure(figsize=(12, 6))

    # -----------------------------------------
    # QAT results
    # -----------------------------------------

    bars = plt.bar(
        x,
        qat_mse,
        width=0.5,
        label="QAT"
    )
    
    plt.bar_label(
        bars,
        fmt="%.4f",
        padding=3
    )

    # -----------------------------------------
    # FP baseline
    # -----------------------------------------

    plt.axhline(
        y=fp_mse,
        linestyle="--",
        linewidth=2,
        label=f"FP baseline ({fp_mse:.4f})"
    )

    # -----------------------------------------
    # PQT baseline
    # -----------------------------------------

    plt.axhline(
        y=pqt_mse,
        linestyle=":",
        linewidth=2,
        label=f"PQT baseline ({pqt_mse:.4f})"
    )

    # -----------------------------------------
    # Labels
    # -----------------------------------------

    plt.xticks(
        x,
        schedules,
        rotation=20
    )

    plt.xlabel("Scheduling Method")
    plt.ylabel("Test MSE")

    plt.title(
        "QAT Test MSE Compared with FP and PQT Baselines"
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


def plot_qat_loss_comparison(results, results_dir):
    """
    Compare QAT training loss across all scheduling methods.
    """

    plt.figure(figsize=(12, 6))

    for result in results:

        schedule_type = result["schedule_type"]

        loss_history = result["loss_history_qat"]

        plt.plot(
            loss_history,
            label=schedule_type
        )

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(
        "QAT Training Loss Across Scheduling Methods"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "qat_loss_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_fp_qat_pqt_loss_comparison(fp_pqt_result, qat_result, results_dir):
    """
    Compare FP, QAT and PQT performance in one figure.

    FP and QAT are plotted as training-loss curves.
    PQT is shown as a horizontal line because it is
    evaluated only after FP training.
    """


    fp_loss = fp_pqt_result["loss_history_fp"]
    qat_loss = qat_result["loss_history_qat"]
    pqt_loss = fp_pqt_result["train_mse_pqt"]

    epochs_fp = np.arange(len(fp_loss))
    epochs_qat = np.arange(len(qat_loss))

    plt.figure(figsize=(12, 6))

    # Floating-point training loss
    plt.plot(
        epochs_fp,
        fp_loss,
        label="Floating-Point"
    )

    # QAT training loss
    plt.plot(
        epochs_qat,
        qat_loss,
        label=f"QAT ({qat_result['schedule_type']})"
    )

    # PQT final MSE
    plt.axhline(
        y=pqt_loss,
        linestyle="--",
        label="PQT Train MSE"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(
        "Floating-Point vs QAT vs PQT"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "fp_qat_pqt_comparison.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

def main():

    # =========================================
    # Configuration
    # =========================================

    RESULTS_DIR = "results/complex_model_non_linear"

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    layer_sizes = [4, 32, 32, 16, 1]

    epochs = 20000
    lr = 0.01

    total_bits = 8
    frac_bits = 4

    schedules = [
        "linear",
        "independent_linear",
        "step",
        "warm_up"
    ]

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
            verbose=True
        )

        qat_results.append(result)
    
    # result = run_schedule_experiment(
    #         schedule_type="linear",

    #         X_train=X_train,
    #         X_test=X_test,

    #         y_train=y_train,
    #         y_test=y_test,

    #         layer_sizes=layer_sizes,

    #         epochs=epochs,
    #         lr=lr,

    #         total_bits=total_bits,
    #         frac_bits=frac_bits,

    #         results_dir=RESULTS_DIR
    # )
    # =========================================
    # Create plots
    # =========================================

    for result in qat_results:

        plot_schedule_behavior(
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