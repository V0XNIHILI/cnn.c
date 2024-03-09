#include "tensor.h"
#include "nn.h"

int main(int argc, char *argv[]) {
    // Test softmax

    size_t tensor_dims [] = {3};
    Tensor *input = create_tensor(1, tensor_dims);

    load_tensor_from_file("src/data.bin", input);

    print_tensor(input);
    printf("INPUT ABOVE \n");

    Tensor *output = softmax(input);

    print_tensor(output);

    destroy_tensor(input);
    destroy_tensor(output);

    // Test linear layer

    size_t tensor_dims2 [] = {3};
    Tensor *input2 = create_tensor(1, tensor_dims2);

    load_tensor_from_file("src/data.bin", input2);

    printf("INPUT2 BELOW \n");
    print_tensor(input2);

    size_t tensor_dims3 [] = {4, 3};

    Tensor *weight = create_tensor(2, tensor_dims3);

    load_tensor_from_file("src/weight.bin", weight);

    printf("WEIGHT BELOW \n");
    print_tensor(weight);

    size_t tensor_dims4 [] = {4};

    Tensor *bias = create_tensor(1, tensor_dims4);

    load_tensor_from_file("src/bias.bin", bias);

    printf("BIAS BELOW \n");
    print_tensor(bias);

    Tensor *output2 = linear(input2, weight, bias);

    printf("OUTPUT2 BELOW \n");

    print_tensor(output2);

    size_t weight_indices[] = {1, 2};
    float val = get_tensor_entry_value(weight, weight_indices);
    printf("VAL: %f\n", (double)val);

    size_t input_indices[] = {0, };
    float val2 = get_tensor_entry_value(input2, input_indices);
    printf("VAL2: %f\n", (double)val2);

    // Try relu on inputs2

    Tensor *output3 = relu(input2);

    printf("OUTPUT3 BELOW \n");

    print_tensor(output3);

    destroy_tensor(output3);

    destroy_tensor(input2);
    destroy_tensor(weight);
    destroy_tensor(bias);
}
