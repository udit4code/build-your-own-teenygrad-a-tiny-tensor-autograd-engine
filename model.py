"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - prod
import numpy as np 
def prod(shape):
    # TODO: Multiply together the elements of a shape tuple to get the total number of elements.
    return int(np.prod(shape))

# Step 2 - argsort
import numpy as np 

def argsort(values):
    # TODO: Return the indices that would sort values in ascending order.
    indices = [x for x in range(len(values))]
    result = sorted(indices, key=lambda i: values[i])
    return result

# Step 3 - make_op_enums
from enum import Enum, auto

def make_op_enums():
    class UnaryOps(Enum):
        NEG = auto()
        RELU = auto()
        LOG = auto()
        EXP = auto()
        SQRT = auto()
        SIGMOID = auto()

    class BinaryOps(Enum):
        ADD = auto()
        SUB = auto()
        MUL = auto()
        DIV = auto()
        CMPLT = auto()
        MAX = auto()

    class ReduceOps(Enum):
        SUM = auto()
        MAX = auto()

    class MovementOps(Enum):
        RESHAPE = auto()
        EXPAND = auto()
        PERMUTE = auto()
    return UnaryOps, BinaryOps, ReduceOps, MovementOps

# Step 4 - LazyBuffer
# Our LazyBuffer is the storage abstraction for the entire framework. 
# Every unary op, binary op, reduction, gradient, parameter, and optimizer update will flow through it.
# If you allow NumPy to infer dtypes, then, the behavior of our engine changes depending on how the user wrote the literal. 


# RULE OF THUMB : 
# | Input to `LazyBuffer`                     | `np.asarray` copies? |
# | ----------------------------------------- | -------------------- |
# | Python list                               | Yes                  |
# | Python tuple                              | Yes                  |
# | Scalar                                    | Yes                  |
# | Existing ndarray, same dtype              | No                   |
# | Existing ndarray, dtype conversion needed | Yes                  |

 
class LazyBuffer:
    def __init__(self, np_array):
        # Convert input to a NumPy ndarray. 
        # If we do not specify dtype, it automatically infers and assigns a compatible type.
        # np.asarray() has the contract : "Convert to ndarray if needed, but avoid copying whenever possible."
        self._np = np.asarray(np_array)
        # The explicit int(...) cast turns numpy scalars into plain Python ints so downstream '==' checks and hashing behave normally
        self.shape = tuple([int(d) for d in self._np.shape])
        self.dtype = self._np.dtype 

    # Why ? Because for the edge-case : 
    def __array__(self, dtype=None):
        return np.asarray(self._np, dtype=dtype)

    def __float__(self):
        return float(self._np)

    def __repr__(self):
        return repr(self._np)

    def __str__(self):
        return str(self._np)

# Step 5 - lazybuffer_const
def const(value, shape):
    x = np.full(shape, value, dtype=np.float32)
    return LazyBuffer(x)

LazyBuffer.const = staticmethod(const)

# Step 6 - rand
def rand(shape, seed=None):
    rng = np.random.RandomState(seed)
    return LazyBuffer(rng.random(shape).astype(np.float32))

# Step 7 - lazybuffer_unary_e
# Please Note that unary functions in numpy like np.maximum(...), np.exp(...), np.log(...), np.sqrt(...)
# are elementwise compute operations, not movement operations. 
# They allocate a new output ndarray and write the computed values into it.

# Everytime we do x = np.array([1, 2, 3]) and then, y = np.exp(x);
# creates a brand-new array y containing [e¹, e², e³]. Since the values themselves have changed, NumPy cannot represent the result merely by changing metadata (shape/strides). 
# Therefore x and y do not share storage: np.shares_memory(x, y) = False. 
# Summary : In Numpy, Unary math ops (exp, sqrt, sigmoid, relu, etc.) actually compute new numbers, so they must allocate new storage.

def e(self, op):
    # The original buffer must stay unchanged. So, we stored the value in a new variable.
    out = None
    # In each of these operations, out and self._np have different values. 
    # So, they must have/point to different storage. 
    if op.name == UnaryOps.NEG.name:
        out = -self._np
    elif op.name == UnaryOps.RELU.name:
        out = np.maximum(self._np, 0)
    elif op.name == UnaryOps.LOG.name:
        out = np.log(self._np)
    elif op.name == UnaryOps.EXP.name:
        out = np.exp(self._np)
    elif op.name == UnaryOps.SQRT.name:
        out = np.sqrt(self._np)
    elif op.name == UnaryOps.SIGMOID.name:
        out = 1.0 / (1.0 + np.exp(-self._np))
    else:
        raise ValueError(f"Unsupported unary op: {op}") 
    assert np.shares_memory(out, self._np) == False, f"For unary elementwise op {op}, we have output and self._np sharing same storage"
    return LazyBuffer(out)
    
LazyBuffer.e = e

# Step 8 - lazybuffer_binary_e
def lazybuffer_binary_e(self, op, other):
    assert isinstance(other, LazyBuffer) , f"other {other} is not a lazy buffer for op {op} with self {self}"
    a = self._np
    b = other._np
    if op.name == BinaryOps.ADD.name:
        output = a + b
    elif op.name == BinaryOps.SUB.name:
        output = a - b
    elif op.name == BinaryOps.MUL.name:
        output = a * b
    elif op.name == BinaryOps.DIV.name:
        output = a / b
    elif op.name == BinaryOps.CMPLT.name:
        # Forgetting to cast the CMPLT boolean result back to a numeric dtype will 
        # break any later arithmetic that consumes the comparison output.
        output = (a < b).astype(a.dtype)
    elif op.name == BinaryOps.MAX.name:
        output = np.maximum(a, b)
    else:
        raise ValueError(f"Unsupported binary op: {op}")
    assert np.shares_memory(output, self._np) == False, f"out {output} and self._np {self._np} sharing same memory, which is not expected for binary op {op} between {self} and {other}"
    assert np.shares_memory(output, other._np) == False, f"out {output} and other._np {other._np} sharing same memory, which is not expected for binary op {op} between {self} and {other}"
    return LazyBuffer(output)


LazyBuffer.e = lazybuffer_binary_e

# Step 9 - lazybuffer_r
# Reductions like sum and max also create new storage and do not share memory with the input. 
# The reason is that a reduction is not merely reinterpreting existing bytes (like reshape or permute); it is computing entirely new values. 
# For example, reducing a (2,3) ndarray with sum(axis=1) produces a (2,1) ndarray whose elements (6, 15, etc.) never existed in the original buffer. 
# Therefore NumPy must allocate a new output array to hold these computed results. 
# In our teenygrad, r(SUM, axis) and r(MAX, axis) should therefore behave like unary and binary compute ops: create a fresh ndarray and wrap it in a new LazyBuffer, with no storage sharing with the input.

def r(self, op, axis):
    if op.name == ReduceOps.SUM.name:
        output = np.sum(self._np, axis=axis, keepdims=True)
    elif op.name == ReduceOps.MAX.name:
        output = np.max(self._np, axis=axis, keepdims=True)
    else:
        raise ValueError(f"Unsupported reduce op: {op}")
    assert np.shares_memory(output, self._np) == False, f"reduce op {op.name} for {self} leads to an output that does not share same memory"
    return LazyBuffer(output)

LazyBuffer.r = r

# Step 10 - lazybuffer_reshape
# reshape is fundamentally different from unary, binary, and reduction operations because it does not compute new values. 
# It only changes how the same block of memory is interpreted. 
# For example, a ndarray with shape (2, 3) and values [1, 2, 3, 4, 5, 6] can be reshaped to (3, 2) without moving or recomputing any data. 
# The bytes in memory stay exactly where they are. 
# Therefore, NumPy typically implements reshape by creating a new ndarray object with different shape/stride metadata that points to the same underlying storage. 

# In our teenygrad, reshape should return a new LazyBuffer, but that new buffer will usually share storage with the original buffer because it is a movement op, not a compute op. The key idea is: same values, same memory, different view of that memory.

def reshape(self, new_shape):
    output = self._np.reshape(new_shape)
    assert np.shares_memory(output, self._np), f"reshape on {self} to {new_shape} leads to an output {output} with different backend storage"
    return LazyBuffer(output)

# Step 11 - lazybuffer_expand
# expand is the interesting edge case. 
# Conceptually, it is a movement operation because it does not compute new values—it only changes how existing values are viewed. 
# For example, a ndarray of shape (3,1) can be expanded to (3,4) by treating each value in the size-1 dimension as if it were repeated four times. 
# NumPy's broadcast_to implements this by creating a view that shares storage with the original array using zero strides along the expanded axes. 
# No new numerical data is computed.

# For example, consider the snippet in numpy : 
# x = np.array([[10],[20],[30]])
# y = np.broadcast_to(x, (3,4)) 
# Now, a stride tells Numpy how many bytes to jump in memory when an index along an axis increases by 1. 
# So, x has shape (3, 1) and as int64 occupies 8 bytes, the strides are x.strides = (8, 8). 
# That means, for x, moving from x[0,0] to x[1,0] jumps 8 bytes. 
# And, moving from x[0, 0] to x[0, 1] would also jump 8 bytes (assuming column 1 existed). 

# Via broadcasting, numpy wants all of y[0,0], y[0,1], y[0,2], y[0,3] to read the same value 10. 
# So, it sets strides for y to y.strides = (8, 0). 
# So, the second stride is 0 bytes. 
# So, when we move from y[0, 0] to y[0, 1], the jump in memory is 0 bytes. 
# Increasing the column index doesn't move anywhere in memory. Every column in that row reads from the same underlying location. 
# That's the entire trick behind broadcasting: NumPy creates the illusion of repeated values by manipulating strides instead of allocating and storing actual copies.


# But, in our implementation, we were explicitly asked to do : np.array(np.broadcast_to(self._np, shape)) 
# The np.array(...) forces a copy. 
# So although expand is conceptually a movement op, our teenygrad implementation materializes the broadcasted values into new storage. 
# After that copy, the expanded buffer no longer shares memory with the original buffer.

def expand(self, new_shape):
    shape = tuple([int(d) for d in new_shape])
    out = np.array(np.broadcast_to(self._np, shape))
    return LazyBuffer(out)

# Step 12 - lazybuffer_permute
# In NumPy, transpose (and therefore our permute) is a classic view operation. When we do:
# x = np.arange(24).reshape(2,3,4)
# y = x.transpose((0,2,1))
# NumPy does not rearrange or copy the underlying numbers. Instead, it creates a new ndarray object y with a different shape and different strides. 
# The data buffer remains the same. Under the hood, the shape changes from (2, 3, 4) to (2, 4, 3) and hence, 
# the strides are permuted accordingly. Conceptually, NumPy is saying: "Read the same bytes, but interpret axis 1 as axis 2 and axis 2 as axis 1."

# Here, in our teenygrad, the behaviour is essentially the same.
# output = self._np.transpose(order) returns a NumPy view that shares storage with self._np. 
# Then, LazyBuffer(output) creates a new LazyBuffer object, but its underlying ndarray typically points to the same storage as the original buffer.


def permute(self, order):
    output = self._np.transpose(order)
    assert np.shares_memory(output, self._np) == True, f"permute({order}) on {self} leads to an output {output}, whose storage is not same as self._np {self._np}"
    return LazyBuffer(output)

# Step 13 - Function
# In our teenygrad engine, the computational graph is typically : 
# Tensor --> Function --> Tensor --> Function --> Tensor 

# The Tensor objects hold the actual values (data, gradients, shape, etc.), while the Function objects represent operations such as Add, Mul, Relu, MatMul, etc. 
# When an operation runs, the Function node remembers its input tensors (parents) and produces an output tensor. 
# Later, during backpropagation, the output tensor follows its attached Function node, and the Function node knows how to route gradients back to the appropriate parent tensors by applying the chain rule.

# We can think of a Function node as a relay/router with math knowledge. 
# It doesn't just connect tensors; it also knows how to transform gradients. 
# For example, an Add node routes the incoming gradient unchanged to both parents, while a Mul node routes gradients as grad * y to x and grad * x to y. 
# The Tensor stores the value; the Function stores the operation and gradient-routing logic. 
# Together they form the autograd graph.

class Function:
    def __init__(self, *tensors):
        # Each element unpacked by tensors is an object of instance Tensor. 
        # We need the param needs_input_grad.
        # Why ? needs_input_grad = [a.requires_grad, b.requires_grad, c.requires_grad] 
        # This is stored, so that later, during backward() pass, an operation/function can decide which gradients need to be computed and returned. 
        # Say, in forward pass, z = x + y, with x.requires_grad = True and y.requires_grad = True.
        # Now, in backward pass, we go from z to x and y via operation node Add. 
        # The Add Node operation/function stores needs_input_grad = [True, False] during forward pass. 
        # So, during backward pass, it is able to deduce that gradient for x needs to be computed (True), and gradient for y needs to be skipped (False).
        self.needs_input_grad = [t.requires_grad for t in tensors]

        # The flag self.requires_grad answers the question : will the output tensor participate in backpropagation ? 
        # Now, in the same example as above, the output of Add Node Function, which is tensor z, has z.requires_grad = True 
        # Why ? Because, at least one of its parent tensors needs gradients. 
        if any(flag is True for flag in self.needs_input_grad):
            self.requires_grad = True
        elif None in self.needs_input_grad:
            # This is weird edge case, when self.needs_input_grad is somehow set to None.
            self.requires_grad = None
        else:
            # The case when none of the parent tensors of current Operation Node has requires_grad=True. 
            self.requires_grad = False

        # This step is crucial. Because it marks a graph edge. 
        # So, since self.requires_grad is True, then for Add node, its output z must have some gradient. 
        # So, it must have at least 1 parent, for gradients to be propagated back to them from output tensor z via Add Node operation. 
        if self.requires_grad is True:
            self.parents = tensors
        # Obvious scenario : if output tensor z of Add Node doesn't need gradient, then, is there any point in remembering its parents? 
        # No, and hence, in this way, we minimise memory by cutting down unnecessary book-keeping of gradients.

# Step 14 - function_forward_backward_stubs (not yet solved)
# TODO: implement

# Step 15 - apply (not yet solved)
# TODO: implement

# Step 16 - Neg (not yet solved)
# TODO: implement

# Step 17 - Relu (not yet solved)
# TODO: implement

# Step 18 - Log (not yet solved)
# TODO: implement

# Step 19 - Exp (not yet solved)
# TODO: implement

# Step 20 - Sqrt (not yet solved)
# TODO: implement

# Step 21 - Sigmoid (not yet solved)
# TODO: implement

# Step 22 - Add (not yet solved)
# TODO: implement

# Step 23 - Sub (not yet solved)
# TODO: implement

# Step 24 - Mul (not yet solved)
# TODO: implement

# Step 25 - Div (not yet solved)
# TODO: implement

# Step 26 - sum_function_forward (not yet solved)
# TODO: implement

# Step 27 - sum_function_backward (not yet solved)
# TODO: implement

# Step 28 - max_function_forward (not yet solved)
# TODO: implement

# Step 29 - max_function_backward (not yet solved)
# TODO: implement

# Step 30 - Reshape (not yet solved)
# TODO: implement

# Step 31 - expand_function_forward (not yet solved)
# TODO: implement

# Step 32 - expand_function_backward (not yet solved)
# TODO: implement

# Step 33 - permute_function_forward_backward (not yet solved)
# TODO: implement

# Step 34 - Tensor (not yet solved)
# TODO: implement

# Step 35 - tensor_from_data (not yet solved)
# TODO: implement

# Step 36 - tensor_creation_helpers (not yet solved)
# TODO: implement

# Step 37 - tensor_randn (not yet solved)
# TODO: implement

# Step 38 - build_topological_order (not yet solved)
# TODO: implement

# Step 39 - tensor_backward (not yet solved)
# TODO: implement

# Step 40 - bind_unary_tensor_methods (not yet solved)
# TODO: implement

# Step 41 - broadcasted (not yet solved)
# TODO: implement

# Step 42 - bind_binary_tensor_methods (not yet solved)
# TODO: implement

# Step 43 - bind_movement_tensor_methods (not yet solved)
# TODO: implement

# Step 44 - bind_reduce_tensor_methods (not yet solved)
# TODO: implement

# Step 45 - tensor_mean (not yet solved)
# TODO: implement

# Step 46 - tensor_transpose (not yet solved)
# TODO: implement

# Step 47 - tensor_matmul_2d (not yet solved)
# TODO: implement

# Step 48 - tensor_softmax (not yet solved)
# TODO: implement

# Step 49 - tensor_log_softmax (not yet solved)
# TODO: implement

# Step 50 - sparse_categorical_cross_entropy (not yet solved)
# TODO: implement

# Step 51 - Linear (not yet solved)
# TODO: implement

# Step 52 - MLP (not yet solved)
# TODO: implement

# Step 53 - sgd_step (not yet solved)
# TODO: implement

# Step 54 - zero_grad (not yet solved)
# TODO: implement

# Step 55 - make_toy_digit_dataset (not yet solved)
# TODO: implement

# Step 56 - accuracy (not yet solved)
# TODO: implement

# Step 57 - train_mlp (not yet solved)
# TODO: implement

# Step 58 - evaluate_mlp (not yet solved)
# TODO: implement

