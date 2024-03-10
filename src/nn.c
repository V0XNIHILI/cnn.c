#include "tensor.h"

Tensor *remove_batch_size_if_present_from_3d_tensor(Tensor *t, bool has_batch_dim) {
    assert(t->n_dims == 4);

    if (has_batch_dim) {
        return t;
    } else {
        size_t channels = t->dims[1];
        size_t height = t->dims[2];
        size_t width = t->dims[3];

        t->n_dims = 3;
        free(t->dims);
        t->dims = malloc(3 * sizeof *t->dims);
        assert(t->dims != NULL);

        t->dims[0] = channels;
        t->dims[1] = height;
        t->dims[2] = width;

        return t;
    }
}

// CHECKED
Tensor *remove_batch_size_if_present_from_1d_tensor(Tensor *t, bool has_batch_dim) {
    assert(t->n_dims == 2);

    if (has_batch_dim) {
        return t;
    } else {
        size_t num_entries = t->dims[1];

        t->n_dims = 1;
        free(t->dims);
        t->dims = malloc(1 * sizeof *t->dims);
        assert(t->dims != NULL);
        
        t->dims[0] = num_entries;

        return t;
    }
}

// CHECKED
Tensor *conv_2d(const Tensor *input, const Tensor *weight, const Tensor *bias, size_t stride) {
    assert(input != NULL);
    assert(weight != NULL);
    assert(bias != NULL);

    bool has_batch_dim = input->n_dims == 4;

    assert(has_batch_dim || input->n_dims == 3);

    size_t batch_size = has_batch_dim ? input->dims[0] : 1;
    size_t input_channels = input->dims[0+has_batch_dim];
    size_t input_height = input->dims[1+has_batch_dim];
    size_t input_width = input->dims[2+has_batch_dim];

    size_t output_channels = weight->dims[0];
    size_t weight_input_channels = weight->dims[1];
    size_t kernel_height = weight->dims[2];
    size_t kernel_width = weight->dims[3];
    size_t bias_size = bias->dims[0];

    assert(weight->n_dims == 4);
    assert(input_channels == weight_input_channels);
    assert(bias->n_dims == 1);
    assert(bias_size == output_channels);

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    size_t output_dims[] = {batch_size, output_channels, output_height, output_width};
    Tensor *output = create_tensor(4, output_dims);

    size_t i;
    #pragma omp parallel for private (i)
    for (i = 0; i < output_channels; i++) {
        for (size_t n = 0; n < input_channels; n++) {
            for (size_t l = 0; l < kernel_height; l++) {
                for (size_t m = 0; m < kernel_width; m++) {
                    float current_weight = get_tensor_entry_value(weight, (size_t[]) {i, n, l, m});
                    bool set_bias = n == 0 && l == 0 && m == 0;

                    for (size_t b = 0; b < batch_size; b++) {
                        for (size_t j = 0; j < output_height; j++) {
                            for (size_t k = 0; k < output_width; k++) {
                                size_t current_input_indices[4];

                                if (has_batch_dim) {
                                    current_input_indices[0] = b;
                                    current_input_indices[1] = n;
                                    current_input_indices[2] = j * stride + l;
                                    current_input_indices[3] = k * stride + m;
                                } else {
                                    current_input_indices[0] = n;
                                    current_input_indices[1] = j * stride + l;
                                    current_input_indices[2] = k * stride + m;
                                }

                                size_t current_output_index = get_tensor_entry_index(output, (size_t[]) {b, i, j, k});

                                if (set_bias) {
                                    output->data[current_output_index] = bias->data[i];
                                }

                                output->data[current_output_index] += get_tensor_entry_value(input, current_input_indices) * current_weight;
                            }
                        }
                    }
                }
            }
        }
    }

    return remove_batch_size_if_present_from_3d_tensor(output, has_batch_dim);
}

Tensor *max_pool_2d(const Tensor *input, size_t pool_size, size_t stride) {
    assert(input != NULL);

    bool has_batch_dim = input->n_dims == 4;

    assert(has_batch_dim || input->n_dims == 3);

    size_t batch_size = has_batch_dim ? input->dims[0] : 1;
    size_t input_channels = input->dims[0+has_batch_dim];
    size_t input_height = input->dims[1+has_batch_dim];
    size_t input_width = input->dims[2+has_batch_dim];

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;

    size_t output_dims[] = {batch_size, input_channels, output_height, output_width};
    Tensor *output = create_tensor(4, output_dims);

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < input_channels; i++) {
            for (size_t j = 0; j < output_height; j++) {
                for (size_t k = 0; k < output_width; k++) {
                    size_t current_output_index = get_tensor_entry_index(output, (size_t[]) {b, i, j, k});

                    float max_value = -INFINITY;

                    for (size_t l = 0; l < pool_size; l++) {
                        for (size_t m = 0; m < pool_size; m++) {
                            size_t current_input_indices[4];

                            if (has_batch_dim) {
                                current_input_indices[0] = b;
                                current_input_indices[1] = i;
                                current_input_indices[2] = j * stride + l;
                                current_input_indices[3] = k * stride + m;
                            } else {
                                current_input_indices[0] = i;
                                current_input_indices[1] = j * stride + l;
                                current_input_indices[2] = k * stride + m;
                            }

                            float current_value = get_tensor_entry_value(input, current_input_indices);

                            max_value = max_value > current_value ? max_value : current_value;
                        }
                    }

                    output->data[current_output_index] = max_value;
                }
            }
        }
    }

    return remove_batch_size_if_present_from_3d_tensor(output, has_batch_dim);
}

// CHECKED
Tensor *relu (const Tensor *input) {
    assert(input != NULL);

    Tensor *output = create_tensor(input->n_dims, input->dims);

    size_t num_elements = get_tensor_element_count(input);

    for (size_t i = 0; i < num_elements; i++) {
        output->data[i] = input->data[i] > 0 ? input->data[i] : 0;
    }

    return output;
}

Tensor *conv_relu_max_pool_2d(const Tensor *input, const Tensor *weight, const Tensor *bias, size_t conv_stride, size_t pool_size, size_t pool_stride) {
    Tensor *conv = conv_2d(input, weight, bias, conv_stride);
    Tensor *relu_tensor = relu(conv);
    destroy_tensor(conv);
    Tensor *max_pool = max_pool_2d(relu_tensor, pool_size, pool_stride);
    destroy_tensor(relu_tensor);

    return max_pool;
}

Tensor *flatten(const Tensor *input, bool has_batch_dim) {
    assert(input != NULL);

    if (input->n_dims == 1) {
        return copy_tensor(input);
    }

    if (has_batch_dim) {
        assert(input->n_dims >= 2);
    }

    size_t num_entries = get_tensor_element_count(input) / (has_batch_dim ? input->dims[0] : 1);

    size_t batch_size = has_batch_dim ? input->dims[0] : 1;

    size_t output_dims[] = {batch_size, num_entries};

    Tensor *output = create_tensor(2, output_dims);

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < num_entries; i++) {
            size_t current_input_indices[] = {b, i};
            size_t current_input_index = get_tensor_entry_index(input, current_input_indices);

            output->data[get_tensor_entry_index(output, current_input_indices)] = input->data[current_input_index];
        }
    }

    return remove_batch_size_if_present_from_1d_tensor(output, has_batch_dim);
}

// CHECKED!
Tensor *linear(const Tensor *input, const Tensor *weight, const Tensor *bias) {
    assert(input != NULL);
    assert(weight != NULL);
    assert(bias != NULL);

    bool has_batch_dim = input->n_dims == 2;

    assert(has_batch_dim || input->n_dims == 1);

    size_t batch_size = has_batch_dim ? input->dims[0] : 1;
    size_t input_size = input->dims[0+has_batch_dim];
    size_t output_size = weight->dims[0];
    size_t weight_input_size = weight->dims[1];
    size_t bias_size = bias->dims[0];

    assert(weight->n_dims == 2);
    assert(input_size == weight_input_size);
    assert(bias->n_dims == 1);
    assert(bias_size == output_size);

    size_t output_dims[] = {batch_size, output_size};

    Tensor *output = create_tensor(2, output_dims);

    for (size_t b = 0; b < batch_size; b++) {
        for (size_t i = 0; i < output_size; i++) {
            size_t current_output_index = get_tensor_entry_index(output, (size_t[]) {b, i});

            output->data[current_output_index] = bias->data[i];

            for (size_t j = 0; j < input_size; j++) {
                // In the case of a 1D tensor, the batch dimension is not present
                // Also, in this case the second index will be ignored
                output->data[current_output_index] += get_tensor_entry_value(input, (size_t []) {has_batch_dim ? b : j, j}) * get_tensor_entry_value(weight, (size_t []) {i, j});
            }
        }
    }

    return remove_batch_size_if_present_from_1d_tensor(output, has_batch_dim);
}

// CHECKED!
Tensor *softmax(const Tensor *input) {
    assert(input != NULL);

    bool has_batch_dim = input->n_dims == 2;

    assert(has_batch_dim || input->n_dims == 1);

    size_t batch_size = has_batch_dim ? input->dims[0] : 1;
    size_t input_size = input->dims[0+has_batch_dim];

    size_t output_dims [] = {batch_size, input_size};

    Tensor *output = create_tensor(2, output_dims);

    for (size_t b = 0; b < batch_size; b++) {
        float sum = 0;

        for (size_t i = 0; i < input_size; i++) {
            size_t current_input_index = get_tensor_entry_index(output, (size_t[]) {b, i});

            float exp_value = (float) exp((double) input->data[current_input_index]);
            output->data[current_input_index] = exp_value;
            sum += exp_value;
        }

        for (size_t i = 0; i < input_size; i++) {
            size_t current_input_index = get_tensor_entry_index(output, (size_t[]) {b, i});

            output->data[current_input_index] /= sum;
        }
    }

    return remove_batch_size_if_present_from_1d_tensor(output, has_batch_dim);
}
