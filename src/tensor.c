#include "tensor.h"

// CHECKED
Tensor *create_tensor(size_t n_dims, const size_t *dims) {
    assert(dims != NULL);

    Tensor *t = malloc(sizeof *t);
    assert(t != NULL);

    size_t num_elements = 1;

    for (size_t i = 0; i < n_dims; i++) {
        num_elements *= dims[i];
    }

    t->data = malloc(num_elements * sizeof *t->data);
    assert(t->data != NULL);

    t->n_dims = n_dims;
    t->dims = malloc(n_dims * sizeof *t->dims);
    assert(t->dims != NULL);

    for (size_t i = 0; i < n_dims; i++) {
        t->dims[i] = dims[i];
    }

    return t;
}

// CHECKED
void destroy_tensor(Tensor *t) {
    assert(t != NULL);

    free(t->dims);
    free(t->data);
    free(t);
}

// CHECKED
size_t get_tensor_element_count(const Tensor *t) {
    assert(t != NULL);

    size_t num_elements = 1;

    for (size_t i = 0; i < t->n_dims; i++) {
        num_elements *= t->dims[i];
    }

    return num_elements;
}

Tensor *copy_tensor(const Tensor *t) {
    assert(t != NULL);

    Tensor *copy = create_tensor(t->n_dims, t->dims);

    size_t num_elements = get_tensor_element_count(t);

    for (size_t i = 0; i < num_elements; i++) {
        copy->data[i] = t->data[i];
    }

    return copy;
}

// CHECKED
void load_tensor_from_file(const char *filename, Tensor *t) {
    assert(filename != NULL);
    assert(t != NULL);

    FILE *file = fopen(filename, "rb");
    assert(file != NULL);

    size_t num_elements = get_tensor_element_count(t);

    for (size_t i = 0; i < num_elements; i++) {
        fread(&t->data[i], sizeof(float), 1, file);  // Use fread for binary reading
    }

    fclose(file);
}

Tensor *create_tensor_from_file(const char *filename, size_t n_dims, const size_t *dims) {
    assert(filename != NULL);

    Tensor *t = create_tensor(n_dims, dims);

    load_tensor_from_file(filename, t);

    return t;
}

// write tensor to file function
void write_tensor_to_file(const char *filename, const Tensor *t) {
    assert(filename != NULL);
    assert(t != NULL);

    FILE *file = fopen(filename, "wb");
    assert(file != NULL);

    size_t num_elements = get_tensor_element_count(t);

    for (size_t i = 0; i < num_elements; i++) {
        fwrite(&t->data[i], sizeof(float), 1, file);  // Use fwrite for binary writing
    }

    fclose(file);
}

// CHECKED
void print_tensor(const Tensor *t) {
    assert(t != NULL);

    size_t num_elements = get_tensor_element_count(t);

    printf("tensor(shape=(");
    
    for (size_t i = 0; i < t->n_dims; i++) {
        printf("%zu", t->dims[i]);

        if (i < t->n_dims - 1) {
            printf(", ");
        }
    }

    printf("), data=[\n");

    for (size_t i = 0; i < num_elements; i++) {
        float current_value = t->data[i];

        printf("  ");

        if (current_value >= 0) {
           printf(" ");
        }

        printf("%f\n", (double) current_value);

        if (t->n_dims > 1 && i != num_elements - 1) {
            size_t dim_product = 1;
    
            for (size_t j = 0; j < t->n_dims - 1; j++) {
                dim_product *= t->dims[t->n_dims-j-1];

                if (i % dim_product == dim_product-1) {
                    printf("   --------\n");
                }
            }
        }
    }

    printf("])\n");
}

// CHECKED; follows the numpy convention for .flatten()
size_t get_tensor_entry_index(const Tensor *t, const size_t *indices) {
    assert(t != NULL);
    assert(indices != NULL);

    size_t index = 0;
    size_t multiplier = 1;

    for (size_t i = 0; i < t->n_dims; i++) {
        size_t reverse_index = t->n_dims - 1 - i;

        index += indices[reverse_index] * multiplier;
        multiplier *= t->dims[reverse_index];
    }

    return index;
}

// CHECKED
float get_tensor_entry_value(const Tensor *t, const size_t *indices) {
    assert(t != NULL);
    assert(indices != NULL);

    return t->data[get_tensor_entry_index(t, indices)];
}

Tensor *add_tensors(const Tensor *a, const Tensor *b) {
    assert(a != NULL);
    assert(b != NULL);

    assert(a->n_dims == b->n_dims);

    for (size_t i = 0; i < a->n_dims; i++) {
        assert(a->dims[i] == b->dims[i]);
    }

    Tensor *output = create_tensor(a->n_dims, a->dims);

    size_t num_elements = get_tensor_element_count(a);

    for (size_t i = 0; i < num_elements; i++) {
        output->data[i] = a->data[i] + b->data[i];
    }

    return output;
}
