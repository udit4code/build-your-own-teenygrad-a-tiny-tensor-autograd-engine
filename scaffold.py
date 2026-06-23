"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Scaffold for: Build Your Own teenygrad -- A Tiny Tensor Autograd Engine.

Imports the full surface of what the student builds in `solution`, then runs a
minimal end-to-end path: make a toy digit dataset, train a small MLP with SGD
and cross entropy, and evaluate held-out accuracy.
"""

import numpy as np

import solution

# The student's `solution` module may name some internal helpers slightly
# differently than the canonical surface list below. Bind every name that
# actually exists so the scaffold never crashes at import time over an
# optional helper that the end-to-end path does not use.
_SURFACE = [
    "prod",
    "argsort",
    "make_op_enums",
    "LazyBuffer",
    "lazybuffer_const",
    "rand",
    "lazybuffer_unary_e",
    "lazybuffer_binary_e",
    "lazybuffer_r",
    "lazybuffer_reshape",
    "lazybuffer_expand",
    "lazybuffer_permute",
    "Function",
    "function_forward_backward_stubs",
    "apply",
    "Neg",
    "Relu",
    "Log",
    "Exp",
    "Sqrt",
    "Sigmoid",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "sum_function_forward",
    "sum_function_backward",
    "max_function_forward",
    "max_function_backward",
    "Reshape",
    "expand_function_forward",
    "expand_function_backward",
    "permute_function_forward_backward",
    "Tensor",
    "tensor_from_data",
    "tensor_creation_helpers",
    "tensor_randn",
    "build_topological_order",
    "tensor_backward",
    "bind_unary_tensor_methods",
    "broadcasted",
    "bind_binary_tensor_methods",
    "bind_movement_tensor_methods",
    "bind_reduce_tensor_methods",
    "tensor_mean",
    "tensor_transpose",
    "tensor_matmul_2d",
    "tensor_softmax",
    "tensor_log_softmax",
    "sparse_categorical_cross_entropy",
    "Linear",
    "MLP",
    "sgd_step",
    "zero_grad",
    "make_toy_digit_dataset",
    "accuracy",
    "train_mlp",
    "evaluate_mlp",
]

for _name in _SURFACE:
    if hasattr(solution, _name):
        globals()[_name] = getattr(solution, _name)

# Names that the end-to-end path below relies on directly.
_REQUIRED = [
    "make_toy_digit_dataset",
    "tensor_from_data",
    "MLP",
    "sparse_categorical_cross_entropy",
    "accuracy",
    "train_mlp",
    "evaluate_mlp",
]
_missing = [n for n in _REQUIRED if n not in globals()]
if _missing:
    raise ImportError(f"solution is missing required names: {_missing}")


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
    if hasattr(value, "lb") and hasattr(value.lb, "np_array"):
        return float(np.asarray(value.lb.np_array))
    lb = getattr(value, "lazydata", None)
    if lb is not None and hasattr(lb, "_np"):
        return float(np.asarray(lb._np))
    if hasattr(value, "numpy"):
        return float(np.asarray(value.numpy()))
    return float(value)


def main():
    """Run the toy digit MLP training pipeline end to end."""
    np.random.seed(0)

    # --- Data preparation -------------------------------------------------
    X_train, y_train = make_toy_digit_dataset(num_samples=200, seed=0)
    X_test, y_test = make_toy_digit_dataset(num_samples=60, seed=1)
    print("train features shape:", np.asarray(X_train).shape)
    print("train labels shape:  ", np.asarray(y_train).shape)
    print("first few labels:    ", np.asarray(y_train).reshape(-1)[:10])

    # --- Sanity check: a single forward + backward -----------------------
    X_tensor = tensor_from_data(X_train, requires_grad=False)
    probe = MLP(in_features=np.asarray(X_train).shape[1], hidden=16, out_features=10, seed=0)
    logits = probe.__call__(X_tensor) if hasattr(probe, "__call__") else probe.forward(X_tensor)
    initial_loss = sparse_categorical_cross_entropy(logits, y_train)
    initial_acc = accuracy(_logits_array(logits), y_train)
    print("initial loss:    ", _scalar(initial_loss))
    print("initial accuracy:", float(initial_acc))

    # --- Training ---------------------------------------------------------
    model, loss_curve = None, None
    result = train_mlp(X_train, y_train, epochs=30, learning_rate=0.1, hidden=16, seed=0)
    if isinstance(result, tuple) and len(result) == 2:
        model, loss_curve = result
    else:
        loss_curve = result

    if loss_curve is not None:
        curve = [float(v) for v in loss_curve]
        print("loss[0], loss[-1]:", round(curve[0], 4), round(curve[-1], 4))
        print("loss decreased:   ", curve[-1] < curve[0])

    # --- Evaluation -------------------------------------------------------
    if model is not None:
        test_acc = evaluate_mlp(model, X_test, y_test)
        print("test accuracy:   ", float(test_acc))


if __name__ == "__main__":
    main()
