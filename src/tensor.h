#ifndef TENSOR_H
#define TENSOR_H

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

typedef struct Tensor Tensor;

struct Tensor {
    size_t n_dims;
    size_t *dims;
    float *data;
};

Tensor *create_tensor(size_t n_dims, const size_t *dims);

void destroy_tensor(Tensor *t);

size_t get_tensor_element_count(const Tensor *t);

Tensor *copy_tensor(const Tensor *t);

void load_tensor_from_file(const char *filename, Tensor *t);

Tensor *create_tensor_from_file(const char *filename, size_t n_dims, const size_t *dims);

void write_tensor_to_file(const char *filename, const Tensor *t);

void print_tensor(const Tensor *t);

size_t get_tensor_entry_index(const Tensor *t, const size_t *indices);

float get_tensor_entry_value(const Tensor *t, const size_t *indices);

Tensor *add_tensors(const Tensor *a, const Tensor *b);

#endif
