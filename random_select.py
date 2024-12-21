#! /usr/bin/python3

from snapshot import SnapshotCallbackBuilder

from sklearn.metrics import balanced_accuracy_score, f1_score

from fitness import *
from diversity_utils import *
import tensorflow as tf
import os.path
import random
import numpy as np
import json
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import uuid
identifier = str(uuid.uuid4())

import argparse

parser = argparse.ArgumentParser(description='Train model with fault params')
parser.add_argument('--dataset', type=str, choices=['mnist', 'cifar10', 'gtsrb', 'pneumonia'], default='cifar10')
parser.add_argument('--fault_amount_min', type=int, choices=[10, 30, 50], default=10)
parser.add_argument('--fault_amount_max', type=int, choices=[10, 30, 50], default=50)
parser.add_argument('--fault_type', type=str, choices=['label_err', 'remove', 'repeat'], default="label_err")
parser.add_argument('--natural', action='store_true')
parser.add_argument('--acc_metric', type=str, choices=['accuracy', 'balanced_accuracy', 'f1'], default='balanced_accuracy')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--partition', type=float, default=63)
parser.add_argument('--snapshots', type=int, default=3)
parser.add_argument('--alpha_zero', type=float, default=0.01)

args = parser.parse_args()



def arch(dataset, model_name, final_fault, symmetric, num_epochs, batch_size):
    (x_train, y_train), (x_test, y_test) = load_training_data(dataset, final_fault, symmetric)

    model = get_trained_model(dataset, model_name, x_train, y_train, num_epochs, batch_size)

    scores = model.evaluate(x_test,
                            y_test,
                            batch_size=batch_size,
                            verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    print("\n\nTraining finished\n\n")

    div_op = "arch"
    write_preds_file(model, x_test, div_op, dataset, model_name, final_fault, symmetric, "0")


def data(dataset, model_name, final_fault, symmetric, num_epochs, batch_size, partition, identifier):
    (x_train, y_train), (x_test, y_test) = load_training_data(dataset, final_fault, symmetric, True, partition)

    model = get_trained_model(dataset, model_name, x_train, y_train, num_epochs, batch_size)

    scores = model.evaluate(x_test,
                            y_test,
                            batch_size=batch_size,
                            verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    print("\n\nTraining finished\n\n")

    div_op = "data"
    write_preds_file(model, x_test, div_op, dataset, model_name, final_fault, symmetric, identifier)


def snapshot(dataset, model_name, final_fault, symmetric, num_epochs, batch_size, num_snapshots, alpha_zero):
    (x_train, y_train), (x_test, y_test) = load_training_data(dataset, final_fault, symmetric)

    M = num_snapshots # number of snapshots
    T = num_epochs # number of epochs
    # initial learning rate is alpha_zero
    model_prefix = 'Model_'

    snapshot = SnapshotCallbackBuilder(T, M, x_test, dataset, model_name, final_fault, symmetric, alpha_zero)

    input_shape = x_train.shape[1:]
    model = get_model_by_name(model_name, input_shape)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['acc'])
    model.fit(x_train, y_train, callbacks=snapshot.get_callbacks(model_prefix=model_prefix), epochs=num_epochs, batch_size=batch_size)


def random_select(encoded_arr, total_models, ens_size, num_div_ops):
    total_arr_len = total_models * (1 + (num_div_ops - 1))
    encoded_arr_seed = [0] * total_arr_len
    ens_size = 3

    total_pos_comb = total_models * (1 + (num_div_ops - 1) * ens_size)
    random_idx_list = random.sample(range(total_pos_comb), ens_size)
    random_idx_list.sort()

    for idx in random_idx_list:
        if idx < total_models:
            encoded_arr[idx] = 1
        if idx >= total_models:
            new_idx = (idx - total_models) // ens_size + total_models
            encoded_arr[new_idx] += 1

    print("Sum random select arr: ", sum(encoded_arr))
    print("Random select arr: ", encoded_arr)

    return encoded_arr

def config_exists(dataset, model_name, final_fault, symmetric, div_op, identifier=""):
    if final_fault == "golden":
        fault_type = golden
    else:
        fault_type = final_fault.split('-')[0]

    if div_op == "arch":
        identifier = "0"

    if symmetric:
        filepath = "./injection/" +  dataset + "/" + fault_type + "/" + div_op + "-" + model_name + "-" + final_fault + "-" + identifier
    else:
        filepath = "./injection/" +  dataset + "/" + fault_type + "/" + div_op + "-" + model_name + "-" + final_fault + "-asymmetric-" + identifier
    return os.path.exists(filepath)


def get_pred_filename(dataset, model_name, final_fault, symmetric, div_op, identifier=""):
    if final_fault == "golden":
        fault_type = golden
    else:
        fault_type = final_fault.split('-')[0]

    if div_op == "arch":
        identifier = "0"

    if symmetric:
        filepath = "./injection/" +  dataset + "/" + fault_type + "/" + div_op + "-" + model_name + "-" + final_fault + "-" + identifier
    else:
        filepath = "./injection/" +  dataset + "/" + fault_type + "/" + div_op + "-" + model_name + "-" + final_fault + "-asymmetric-" + identifier
    return filepath


def is_valid_encoding(encoded_arr, total_models, ens_size, num_div_ops):
    return (len(encoded_arr) == total_models * num_div_ops and
            sum(encoded_arr) == ens_size)


def decode(encoded_arr, dataset, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero):
    model_list = ["ConvNet", "DeconvNet", "MobileNet", "ResNet18", "ResNet50", "VGG11", "VGG16"]

    ens_size = 3
    N = int(len(encoded_arr)/ens_size)

    pred_filenames = []

    for idx, config in enumerate(encoded_arr):
        if idx < N:
            if config != 0:
                model_name = model_list[idx]
                div_op = "arch"
                if not config_exists(dataset, model_name, final_fault, symmetric, div_op):
                    print("Training arch: ", model_name)
                    arch(dataset, model_name, final_fault, symmetric, num_epochs, batch_size)
                pred_filenames.append(get_pred_filename(dataset, model_name, final_fault, symmetric, div_op))

        elif idx < 2*N:
            if config != 0:
                model_name = model_list[idx-N]
                div_op = "data"
                for j in range(config):
                    identifier = str(j)
                    if not config_exists(dataset, model_name, final_fault, symmetric, div_op, identifier):
                        print("Training data div: ", model_name)
                        data(dataset, model_name, final_fault, symmetric, num_epochs, batch_size, partition, identifier)
                    pred_filenames.append(get_pred_filename(dataset, model_name, final_fault, symmetric, div_op, identifier))

        elif idx < 3*N:
            if config != 0:
                model_name = model_list[idx-2*N]
                div_op = "snapshot"
                if not config_exists(dataset, model_name, final_fault, symmetric, div_op, str(ens_size-1)):
                    print("Training snapshots: ", model_name)
                    snapshot(dataset, model_name, final_fault, symmetric, num_epochs, batch_size, ens_size, alpha_zero)
                for j in range(config):
                    identifier = str(j)
                    pred_filenames.append(get_pred_filename(dataset, model_name, final_fault, symmetric, div_op, identifier))

    return pred_filenames


def read_ground_golden_labels(dataset):
    groundtruth_filename = "./groundtruth/" + dataset
    with open(groundtruth_filename, "r") as f:
        ground_labels = json.load(f)

    golden_filename = "./golden/unified-" + dataset + "-golden.txt"
    with open(golden_filename, "r") as f:
        golden_labels = json.load(f)

    return ground_labels, golden_labels


def eval_ensemble(encoded_arr, dataset, ground_labels, golden_labels, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero, acc_metric):
    encoded_str = ''.join(map(str, encoded_arr))
    ens_size = sum(encoded_arr)

    if final_fault == "golden":
        fault_type = golden
        fault_amt = "0"
    else:
        fault_type = final_fault.split('-')[0]
        fault_amt = final_fault.split('-')[1]

    if symmetric:
        ens_fprefix = "./injection/" +  dataset + "/" + fault_type + "/" + final_fault + "-" + encoded_str
    else:
        ens_fprefix = "./injection/" +  dataset + "/" + fault_type + "/" + final_fault + "-asymmetric-" + encoded_str

    ens_fname_pred = ens_fprefix + "-pred.csv"
    ens_fname_corr = ens_fprefix + "-correct.csv"

    if not os.path.exists(ens_fname_pred) or not os.path.exists(ens_fname_corr):
        pred_filenames = decode(encoded_arr, dataset, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero)
        write_ensemble_decision(encoded_arr, pred_filenames, dataset, final_fault, symmetric)

    #Directly compute using ens_fname
    curr_df = pd.read_csv(ens_fname_pred)
    df = curr_df.copy()
    curr_df["ground"] = ground_labels

    # Calculate accuracy
    corr = curr_df[curr_df["ens"]==curr_df["ground"]].shape[0]
    total = curr_df.shape[0]

    if acc_metric == "balanced_accuracy":
        accuracy = balanced_accuracy_score(curr_df["ground"], curr_df["ens"]) # Balanced Accuracy
    elif acc_metric == "f1":
        accuracy = f1_score(curr_df["ground"], curr_df["ens"], average="binary") # Macro F1
    else:
        accuracy = corr / total # Regular Accuracy

    if fault_amt != "0":
        curr_df = curr_df.iloc[golden_labels]
        incorr = curr_df[curr_df["ens"]!=curr_df["ground"]].shape[0]
        total = curr_df.shape[0]
        ad = incorr / total
    else:
        ad = 0

    resilience = ad

    gdf = pd.read_csv(ens_fname_corr)
    disagreement_measure = get_disagreement_measure(gdf)
    shannon_entropy = get_avg_eh(df, ens_size)
    diversity = 0.5 * disagreement_measure + 0.5 * shannon_entropy

    accuracy = str(round(accuracy, 2))
    resilience = str(round(resilience, 2))
    diversity = str(round(diversity, 2))

    return accuracy, resilience, diversity


def print_best_ensemble(encoded_str, ens_stat_list, dataset, fault_type, elapsed_time, symmetric):
    elapsed_time = str(int(elapsed_time))
    if symmetric:
        desemble_logfilename = "./output/" + dataset + "_" + fault_type + "_random_select"
    else:
        desemble_logfilename = "./output/" + dataset + "_" + fault_type + "_asymmetric_random_select"

    with open(desemble_logfilename, "w") as f:
        for ens_stats in ens_stat_list:
            fault_amount = ens_stats[0]
            accuracy = ens_stats[1]
            resilience = ens_stats[2]
            diversity = ens_stats[3]
            entry = f'{encoded_str}, {fault_type}, {fault_amount}, {accuracy}, {resilience}, {diversity}, {elapsed_time}'
            f.write(entry)
            f.write("\n")


def get_fault_amt_range(fault_amt_lower, fault_amt_higher):
    all_fault_range = [10, 30, 50]
    return [str(fault_amt) for fault_amt in all_fault_range if fault_amt_lower <= fault_amt <= fault_amt_higher]


def main():
    dataset = args.dataset
    fault_type = args.fault_type
    symmetric = not args.natural
    fault_amount_min = args.fault_amount_min
    fault_amount_max = args.fault_amount_max
    acc_metric = args.acc_metric

    partition = args.partition
    num_epochs = args.epochs
    batch_size = args.batch_size
    alpha_zero = args.alpha_zero

    N = total_models = 7
    num_div_ops = 3
    ens_size = 3

    fault_amt_arr = get_fault_amt_range(fault_amount_min, fault_amount_max)

    ground_labels, golden_labels = read_ground_golden_labels(dataset)

    zeros = [0] * N

    encoded_arr = []
    encoded_arr.extend(zeros)
    encoded_arr.extend(zeros)
    encoded_arr.extend(zeros)

    start = time.time()

    encoded_arr = random_select(encoded_arr, total_models, ens_size, num_div_ops)
    encoded_str = ''.join(map(str, encoded_arr))

    ens_stat_list = []

    for fault_amt in fault_amt_arr:
        final_fault = fault_type + "-" + str(fault_amt)
        accuracy, resilience, diversity = eval_ensemble(encoded_arr, dataset, ground_labels, golden_labels, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero, acc_metric)
        ens_stat_list.append([fault_amt, accuracy, resilience, diversity])

    end = time.time()
    elapsed_time = end - start

    print_best_ensemble(encoded_str, ens_stat_list, dataset, fault_type, elapsed_time, symmetric)


if __name__ == "__main__":
    main()

