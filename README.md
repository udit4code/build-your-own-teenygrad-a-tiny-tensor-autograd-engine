# Build Your Own teenygrad: A Tiny Tensor Autograd Engine

Construct a minimal deep learning framework from scratch, layering a lazy numpy-backed buffer, a reverse-mode automatic differentiation engine, a tensor API, and neural network primitives. By the end you train a small MLP on a toy digit dataset, mirroring how real autograd libraries like tinygrad work under the hood.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** prod
- [x] **2.** argsort
- [x] **3.** make_op_enums
- [x] **4.** LazyBuffer
- [x] **5.** lazybuffer_const
- [x] **6.** rand
- [x] **7.** lazybuffer_unary_e
- [x] **8.** lazybuffer_binary_e
- [x] **9.** lazybuffer_r
- [x] **10.** lazybuffer_reshape
- [x] **11.** lazybuffer_expand
- [x] **12.** lazybuffer_permute
- [x] **13.** Function
- [x] **14.** function_forward_backward_stubs
- [x] **15.** apply
- [ ] **16.** Neg
- [ ] **17.** Relu
- [ ] **18.** Log
- [ ] **19.** Exp
- [ ] **20.** Sqrt
- [ ] **21.** Sigmoid
- [ ] **22.** Add
- [ ] **23.** Sub
- [ ] **24.** Mul
- [ ] **25.** Div
- [ ] **26.** sum_function_forward
- [ ] **27.** sum_function_backward
- [ ] **28.** max_function_forward
- [ ] **29.** max_function_backward
- [ ] **30.** Reshape
- [ ] **31.** expand_function_forward
- [ ] **32.** expand_function_backward
- [ ] **33.** permute_function_forward_backward
- [ ] **34.** Tensor
- [ ] **35.** tensor_from_data
- [ ] **36.** tensor_creation_helpers
- [ ] **37.** tensor_randn
- [ ] **38.** build_topological_order
- [ ] **39.** tensor_backward
- [ ] **40.** bind_unary_tensor_methods
- [ ] **41.** broadcasted
- [ ] **42.** bind_binary_tensor_methods
- [ ] **43.** bind_movement_tensor_methods
- [ ] **44.** bind_reduce_tensor_methods
- [ ] **45.** tensor_mean
- [ ] **46.** tensor_transpose
- [ ] **47.** tensor_matmul_2d
- [ ] **48.** tensor_softmax
- [ ] **49.** tensor_log_softmax
- [ ] **50.** sparse_categorical_cross_entropy
- [ ] **51.** Linear
- [ ] **52.** MLP
- [ ] **53.** sgd_step
- [ ] **54.** zero_grad
- [ ] **55.** make_toy_digit_dataset
- [ ] **56.** accuracy
- [ ] **57.** train_mlp
- [ ] **58.** evaluate_mlp

---

Built on Deep-ML.
