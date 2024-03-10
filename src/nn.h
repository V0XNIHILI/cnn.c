#ifndef NN_H
#define NN_H

#include <stdbool.h>
#include <omp.h>

#include "tensor.h"

Tensor *remove_batch_size_if_present_from_3d_tensor(Tensor *t, bool has_batch_dim);

Tensor *remove_batch_size_if_present_from_1d_tensor(Tensor *t, bool has_batch_dim);

Tensor *conv_2d(const Tensor *input, const Tensor *weight, const Tensor *bias, size_t stride);

Tensor *max_pool_2d(const Tensor *input, size_t pool_size, size_t stride);

Tensor *relu(const Tensor *input);

Tensor *conv_relu_max_pool_2d(const Tensor *input, const Tensor *weight, const Tensor *bias, size_t conv_stride, size_t pool_size, size_t pool_stride);

Tensor *flatten(const Tensor *input, bool has_batch_dim);

Tensor *linear(const Tensor *input, const Tensor *weight, const Tensor *bias);

Tensor *softmax(const Tensor *input);

#endif
