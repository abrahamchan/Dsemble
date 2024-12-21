#! /usr/bin/python3

# Script to decode encoded ensembles
# Usage: python decode.py <encoded_str>
# Example: python decode.py 010000000000010100000


import sys

model_list = ["ConvNet", "DeconvNet", "MobileNet", "ResNet18", "ResNet50", "VGG11", "VGG16"]


def is_valid_encoding(encoded_arr, total_models, ens_size, num_div_ops):
    return (len(encoded_arr) == total_models * num_div_ops and
            sum(encoded_arr) == ens_size)


def decode_str(encoded_str):
    return [int(char) for char in list(encoded_str)]


def decode(encoded_arr, ens_size):
    N = int(len(encoded_arr)/ens_size)

    pred_filenames = []

    for idx, config in enumerate(encoded_arr):
        if idx < N:
            if config != 0:
                model_name = model_list[idx]
                pred_filenames.append(model_name)

        elif idx < 2*N:
            if config != 0:
                model_name = model_list[idx-N]
                for j in range(config):
                    identifier = str(j)
                    pred_filenames.append("Data Div Op on " + model_name)

        elif idx < 3*N:
            if config != 0:
                model_name = model_list[idx-2*N]
                for j in range(config):
                    identifier = str(j)
                    pred_filenames.append("Snapshot Div Op on " + model_name)

    return pred_filenames


def main():
    encoded_str = sys.argv[1]
    encoded_arr = decode_str(encoded_str)
    total_models = len(model_list)
    ens_size = 3
    num_div_ops = 3

    if is_valid_encoding(encoded_arr, total_models, ens_size, num_div_ops):
        decoded_arr = decode(encoded_arr, ens_size)
        print("Ensemble combination decoded: ", decoded_arr)
    else:
        print("Invalid ensemble encoding supplied!")


if __name__ == "__main__":
    main()

