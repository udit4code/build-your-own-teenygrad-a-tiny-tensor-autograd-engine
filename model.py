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
    
def make_op_enums():
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

# We are designing, based on the following abstraction. 
# The Tensor is the high-level autograd object, while LazyBuffer is the low-level storage object that actually owns the numerical data. 
# Tensor
# ├── lazydata (LazyBuffer)
# ├── grad
# ├── requires_grad
# ├── _ctx (Function that created it)
# └── autograd metadata

# The Tensor is what users interact with (x + y, x.relu(), x.backward()), but the actual values live inside the LazyBuffer (typically as LazyBuffer._np). 
# During a forward pass, Function.forward() operates on the input tensors' LazyBuffers and produces a new LazyBuffer; then a new Tensor wraps that buffer and records the Function context for backprop. 
# So we can think of Tensor = autograd wrapper, LazyBuffer = numerical storage/backend.


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

# Step 14 - function_forward_backward_stubs
# Right now, Function is acting like an abstract base class : 
# Function
# ├── Add
# ├── Mul
# ├── Relu
# ├── Log
# └── ...

# Every subclass must provide forward(...) and backward(...)
# Why 

def function_forward_backward_stubs():
    def forward(self, *args, **kwargs):
        # Why did we make it type(self).__name__ ? 
        # Because, when Add(Function) class calls Add().forward(...) , 
        # then, in that case, the inherited stub executes, and type(self).__name__ evaluates to "Add" 
        # This helps in better debugging.
        raise NotImplementedError(f"forward not implemented for {type(self).__name__}")

    def backward(self, *args, **kwargs):
        raise NotImplementedError(f"backward not implemented for {type(self).__name__}")

    Function.forward = forward
    Function.backward = backward

    return Function

# Step 15 - apply
# What is happening under the hood ? 
# Say, we write z = Add.apply(x, y)

# Then, internally, ctx = Add(x, y) creates a Function Node object as shown below. 
# x ----\
#        Add(ctx)
# y ----/

# The constructor of Function Node records : 
# ctx.parents = (x, y)
# ctx.needs_input_grad = [...]
# ctx.requires_grad = ...

# Next, ctx.forward(x.lazydata, y.lazydata) runs the actual math on the LazyBuffers of x and y, producing a new buffer.
# Then, out = Tensor(out_buf, requires_grad=ctx.requires_grad) creates the output tensor. 

# Finally, out.__ctx = ctx creates the crucial graph link: 
# x ----\
#        Add(ctx) ---> out
# y ----/

# Notice something subtle : The Function does not attach inputs to outputs. 
# Rather, the output Tensor stores a pointer to the Function that created it, 
# while the Function stores pointers to its parent tensors. 
# This can be summed up by the below diagram : 
# out
#  │
#  ▼
# ctx (Add)
#  │
#  ├── x
#  └── y

# So, the overall picture is : The autograd engine starts at out tensor, then follows out._ctx to reach the Function node. 
# From the Function Node ctx, it follows ctx.parents to continue traversing backward through the graph. 
# This is the core graph wiring that every autograd system uses.




@classmethod
def apply(cls, *tensors, **kwargs):
    # Step 1: Build the Function context
    ctx = cls(*tensors)

    # Step 2: Run forward on the underlying buffers
    out_buf = ctx.forward(
        *[t.lazydata for t in tensors],
        **kwargs
    )

    # Step 3: Wrap the result in a Tensor
    out = Tensor(out_buf,requires_grad=ctx.requires_grad)

    # Step 4: Link graph only when gradients are needed
    if ctx.requires_grad:
        out._ctx = ctx
    return out


# Provided: attaches apply onto the Function base class. Leave this as-is.
for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

# Step 16 - Neg
# Say, we have:
# Tensor x -> Function Node Neg -> Tensor y

# During the forward pass, the Neg Function consumes the underlying
# LazyBuffer of Tensor x and produces a new LazyBuffer: out_buf = LazyBuffer(-x._np)
# This output buffer is then wrapped in Tensor y.
# Mathematically: y = f(x) = -x

# The local derivative of Neg is: dy/dx = -1

# During the backward pass, traversal happens in the reverse direction:
# Tensor x <- Function Node Neg <- Tensor y
# The upstream gradient dL/dy arrives at Neg.backward(...) as the LazyBuffer grad_output.

# By the chain rule:
#     dL/dx = dL/dy * dy/dx
#           = dL/dy * (-1)
#           = -dL/dy

# Here:
#     dL/dy  -> upstream gradient (grad_output)
#     dy/dx  -> local gradient of Neg
#     dL/dx  -> gradient to send to the parent Tensor x

# Since Neg has exactly one parent Tensor x, it returns LazyBuffer(-grad_output._np)
# which numerically represents dL/dx.
#
# The autograd engine will later take this returned gradient
# and accumulate it into Tensor x.grad.

class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)

    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)

# Step 17 - Relu
# Say, we have: Tensor x -> Function Node Relu -> Tensor y

# During the forward pass, the Relu Function consumes the underlying
# LazyBuffer of Tensor x and produces a new LazyBuffer: out_buf = max(x, 0)
# This output buffer is then wrapped in Tensor y.
# Mathematically: y = f(x) = max(x, 0)
# Example:
#     x = [-2, -0.5, 0, 1, 3]
#     y = [ 0,    0, 0, 1, 3]
# Notice that all negative values are clipped to zero,
# while positive values pass through unchanged.


# Why do we store self.ret ?

# During backward propagation, Relu needs to know which
# elements were active (positive) during the forward pass.
# We could store the original input x, but we do not need it.
# The output already contains enough information:   self.ret > 0   <=>   original input was positive

# Therefore we cache: self.ret = relu(x) and reuse it during backward.

# Local derivative of ReLU : ReLU is piecewise:
# relu(x) = 0   if x <= 0 else x   
# Therefore: d(relu)/dx = 0   if x <= 0 else 1 
# Intuition:
# * Negative inputs are "blocked" by ReLU. Their gradients should also be blocked.
# * Positive inputs pass through unchanged. Their gradients should flow unchanged.
#
# Backward pass : During backpropagation:
# Tensor x <- Function Node Relu <- Tensor y
# The upstream gradient dL/dy arrives as grad_output.
# By the chain rule: dL/dx = dL/dy * d(relu)/dx
# The derivative d(relu)/dx is exactly a binary mask: mask = (self.ret > 0)
# Example:
#     self.ret     = [0, 0, 0, 1, 3]
#     mask         = [0, 0, 0, 1, 1]
# Suppose:
#     grad_output = [1, 1, 1, 1, 1]
# Then:
#     grad_input = mask * grad_output
#                = [0, 0, 0, 1, 1]
# Thus gradients only flow through the neurons that were active during the forward pass.

# The returned LazyBuffer numerically represents dL/dx, which the autograd engine will later accumulate into
# the parent Tensor's .grad field.

class Relu(Function):
    def forward(self, x):
        self.ret = e(x, UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        zero = LazyBuffer.const(0, self.ret.shape)
        mask = lazybuffer_binary_e(zero,BinaryOps.CMPLT,self.ret)
        return lazybuffer_binary_e(mask,BinaryOps.MUL,grad_output)

# Step 18 - Log
# Say, we have: Tensor x -> Function Node Log -> Tensor y
# During the forward pass, the Log Function consumes the
# underlying LazyBuffer of Tensor x and produces: y = log(x)
# Example: x = [1, e, e²] and y = [0, 1, 2]
#
# Why do we store self.x ?
# The derivative of log depends on the original input: d(log(x))/dx = 1/x
# Therefore, during backward propagation, we must still have access to the original x values.
#
# We cache: self.x = x during the forward pass.

# Local derivative of Log : For: y = log(x)
# the derivative is: dy/dx = 1/x
# This derivative is computed elementwise.
#
# Backward pass : During backpropagation:
# Tensor x <- Function Node Log <- Tensor y
#
# The upstream gradient dL/dy arrives as grad_output.
# By the chain rule: dL/dx = dL/dy * dy/dx = grad_output * (1/x) = grad_output / x. For example : 
# x           = [1, e, e²]
# grad_output = [1, 1, 1]
# grad_input  = [1, 1/e, 1/e²]
#
# The returned LazyBuffer numerically represents dL/dx, 
# which the autograd engine will later accumulate into
# the parent Tensor's .grad field.

class Log(Function):
    def forward(self, x):
        # Step 1 : Save original input because d(log(x))/dx = 1/x
        # Doubt : Why save x on self.x, when an object of Function class (i.e self) knows about its parent Tensor? 
        # This is because self.backward(...) method doesn't receive parent Tensors. It only receives LazyBuffers
        self.x = x
        # Step 2 : Compute ln(x)
        return e(x, UnaryOps.LOG) 

    # Guided by the math : dL/dx = dL/dy * dy/dx = grad_output * (1/x) = grad_output / x
    def backward(self, grad_output):
        # Please note that the contract of self.backward(...) is LazyBuffer -> LazyBuffer, not Tensor -> Tensor. 
        # So, inside self.backward(...), we need access to the numerical values required for the derivative. 
        # Hence, we need to cache/save the original input values, which turns out to be x.
        # Instead of doing caching for x, we could still fetch it from the parent Tensor, as Function class has reference to it. 
        # We can do something like x = self.parents[0].lazydata to get the original LazyBuffer. 
        # But, despite that, we still cache x on self.x. Why ? 
        # Because, backward should be self-contained and independent of Tensor internals. 
        # Saving exactly what the derivative needs is simpler and faster. 
        # Not every backward formula needs the entire input tensor. 
        # For example : Log needs x, while exp(x) only needs exp(x). 
        # So, for exp(x), we need to save self.ret instead of self.x , while for Log, we need to save self.x 
        # Guiding principle : Save the smallest piece of information needed to compute the local derivative later.
        return lazybuffer_binary_e(grad_output,BinaryOps.DIV, self.x)

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

