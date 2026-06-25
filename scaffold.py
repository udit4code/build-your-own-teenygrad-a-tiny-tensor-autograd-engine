"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold for: Build Your Own teenygrad -- A Tiny Tensor Autograd Engine.

Runs a minimal end-to-end path: make a toy digit dataset, train a small MLP
with SGD and cross entropy, and evaluate held-out accuracy.

Every function the student implements is concatenated ABOVE this scaffold, so
they are already in this module's namespace -- call them directly (there is no
separate `solution` module to import).
"""

import numpy as np


def _logits_array(logits):
    """Best-effort unwrap of a forward output into a plain numpy array."""
    lb = getattr(logits, "lazydata", None)
    if lb is not None and hasattr(lb, "_np"):
        return np.asarray(lb._np)
    if hasattr(logits, "numpy"):
        return np.asarray(logits.numpy())
    return np.asarray(logits)


def _scalar(value):
    """Best-effort unwrap of a (possibly Tensor-wrapped) scalar into a float."""
    lb = getattr(value, "lazydata", None)
    if lb is not None and hasattr(lb, "_np"):
        return float(np.asarray(lb._np))
    if hasattr(value, "numpy"):
        return float(np.asarray(value.numpy()))
    return float(value)


def main():
    """Run the toy digit MLP training pipeline end to end."""
    np.random.seed(0)

    # --- Data ------------------------------------------------------------
    X_train, y_train = make_toy_digit_dataset(num_samples=200, seed=0)
    X_test, y_test = make_toy_digit_dataset(num_samples=60, seed=1)
    print("train features shape:", np.asarray(X_train).shape)
    print("train labels shape:  ", np.asarray(y_train).shape)
    print("first few labels:    ", np.asarray(y_train).reshape(-1)[:10])

    # --- Sanity: one forward + loss --------------------------------------
    n_features = np.asarray(X_train).shape[1]
    n_classes = int(np.asarray(y_train).max()) + 1
    probe = MLP(in_features=n_features, hidden=16, out_features=n_classes, seed=0)
    logits = probe(tensor_from_data(X_train, requires_grad=False))
    initial_loss = sparse_categorical_cross_entropy(logits, y_train)
    initial_acc = accuracy(_logits_array(logits), y_train)
    print("initial loss:    ", _scalar(initial_loss))
    print("initial accuracy:", float(initial_acc))

    # --- Training --------------------------------------------------------
    model, loss_curve = train_mlp(X_train, y_train, epochs=30, learning_rate=0.1, hidden=16, seed=0)
    curve = [float(v) for v in loss_curve]
    print("loss[0], loss[-1]:", round(curve[0], 4), round(curve[-1], 4))
    print("loss decreased:   ", curve[-1] < curve[0])

    # --- Evaluation ------------------------------------------------------
    test_acc = evaluate_mlp(model, X_test, y_test)
    print("test accuracy:   ", float(test_acc))


if __name__ == "__main__":
    main()

