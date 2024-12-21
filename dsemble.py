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
parser.add_argument('--fault_type', type=str, choices=['label_err', 'remove', 'repeat'], default='label_err')
parser.add_argument('--natural', action='store_true')
parser.add_argument('--acc_metric', type=str, choices=['accuracy', 'balanced_accuracy', 'f1'], default='balanced_accuracy')

parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--partition', type=float, default=63)
parser.add_argument('--snapshots', type=int, default=3)
parser.add_argument('--ens_size', type=int, choices=[3, 5, 7], default=3)
parser.add_argument('--alpha_zero', type=float, default=0.01)
parser.add_argument('--crossover_prob', type=float, default=0.8)
parser.add_argument('--mutation_prob', type=float, default=0.5)

parser.add_argument('--max_iterations', type=int, default=10)

args = parser.parse_args()


model_list = ["ConvNet", "DeconvNet", "MobileNet", "ResNet18", "ResNet50", "VGG11", "VGG16"]


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


def contraceptive_correction(encoded_arr, total_models, ens_size, num_div_ops):
    if is_valid_encoding(encoded_arr, total_models, ens_size, num_div_ops):
        return encoded_arr

    extra_elems = sum(encoded_arr) - ens_size

    if extra_elems == 0:
        return encoded_arr

    idx_with_nonzero = []

    if extra_elems > 0:
    # only works if more elements than required
        for i, e in enumerate(encoded_arr):
            for j in range(e):
                idx_with_nonzero.extend([i])
    else:
        idx_with_nonzero = range(len(encoded_arr))

    print("Array before correction: ", encoded_arr)

    random_idx_list = random.sample(idx_with_nonzero, abs(extra_elems))
    random_idx_list.sort()
    for idx in random_idx_list:
        if extra_elems > 0:
            encoded_arr[idx] -= 1
        elif extra_elems < 0 and idx >= total_models:
            encoded_arr[idx] += 1
    return encoded_arr


def crossover(encoded_arrA, encoded_arrB, total_models, ens_size, num_div_ops, crossover_prob, num_crossovers=2):
    random_float = random.uniform(0, 1)
    if crossover_prob < random_float:
        return encoded_arrA, encoded_arrB

    print(encoded_arrA)

    random_idx_list = random.sample(range(1,len(encoded_arrA)-1), num_crossovers)
    random_idx_list.sort()

    crossover_idx = 0

    dont_swap = False

    print("Before encoded_arrA: ", encoded_arrA)
    print("Before encoded_arrB: ", encoded_arrB)

    for idx, config in enumerate(encoded_arrA):
        crossover_point = random_idx_list[crossover_idx]
        if idx < crossover_point and not dont_swap:
            temp = encoded_arrA[idx]
            encoded_arrA[idx] = encoded_arrB[idx]
            encoded_arrB[idx] = temp
        elif idx == crossover_point:
            crossover_idx += 1
            dont_swap ^= 1

        if crossover_idx >= num_crossovers:
            break


    encoded_arrA = contraceptive_correction(encoded_arrA, total_models, ens_size, num_div_ops)
    encoded_arrB = contraceptive_correction(encoded_arrB, total_models, ens_size, num_div_ops)

    print("\nAfter encoded_arrA: ", encoded_arrA)
    print("After encoded_arrB: ", encoded_arrB)
    return encoded_arrA, encoded_arrB


def mutation(encoded_arr, total_models, ens_size, num_div_ops, mutation_prob):
    total_pos_comb = total_models * (1 + (num_div_ops - 1))
    num_elems_mutate = int(mutation_prob * total_pos_comb)
    random_idx_list = random.sample(range(total_pos_comb), num_elems_mutate)
    random_idx_list.sort()

    extra_elems = 0

    print(encoded_arr)

    for idx in random_idx_list:
        if idx < total_models:
            encoded_arr[idx] ^= 1
            if (encoded_arr[idx] == 1):
                extra_elems += 1
            else:
                extra_elems -= 1
        if idx >= total_models:
            if (extra_elems > 0 and encoded_arr[idx] > 0):
                encoded_arr[idx] -= 1
                extra_elems -= 1
            else:
                encoded_arr[idx] += 1
                extra_elems += 1

    contraceptive_correction(encoded_arr, total_models, ens_size, num_div_ops)

    print("\nExtra_elems: ", extra_elems)
    print("Sum mutated: ", sum(encoded_arr))

    print("Mutated arr: ", encoded_arr)

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


def decode_str(encoded_str):
    return [int(char) for char in list(encoded_str)]


def decode(encoded_arr, dataset, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero, ens_size):
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
                if not config_exists(dataset, model_name, final_fault, symmetric, div_op, "2"):
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


def eval_fitness(encoded_arr, dataset, ground_labels, golden_labels, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero, acc_metric):
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
        pred_filenames = decode(encoded_arr, dataset, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero, ens_size)
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

    # Calculate resilience
    if fault_amt != "0":
        curr_df = curr_df.iloc[golden_labels]
        incorr = curr_df[curr_df["ens"]!=curr_df["ground"]].shape[0]
        total = curr_df.shape[0]
        ad = incorr / total
    else:
        ad = 0

    resilience = ad

    # Calculate diversity metrics
    gdf = pd.read_csv(ens_fname_corr)

    shannon_entropy = get_avg_eh(df, ens_size)
    if ens_size == 3:
        disagreement_measure = get_disagreement_measure(gdf)
        diversity = 0.5 * disagreement_measure + 0.5 * shannon_entropy
    else:
        diversity = shannon_entropy

    fitness = 0.5 * (1 - resilience) + 0.5 * diversity

    accuracy = round(accuracy, 2)
    resilience = round(resilience, 2)
    diversity = round(diversity, 2)

    return fitness, accuracy, resilience, diversity


def log(dataset, fault_type, num_fault_amts, best_ensembles, elapsed_time, symmetric):
    elapsed_time = str(int(elapsed_time))
    if symmetric:
        desemble_logfilename = "./output/" + dataset + "_" + fault_type + "_" + identifier + "_" + elapsed_time
    else:
        desemble_logfilename = "./output/" + dataset + "_" + fault_type + "_asymmetric_" + identifier + "_" + elapsed_time

    with open(desemble_logfilename, "w") as f:
        #best_ensembles = {"encoded_str" : ["final_fault", "accuracy", "resilience", "diversity"]}

        for encoded_str, ens_stat_list in best_ensembles.items():
            for i in range(1, num_fault_amts+1):
                ens_stats = ens_stat_list[i]
                fault_amount = ens_stats[1]
                accuracy = ens_stats[2]
                resilience = ens_stats[3]
                diversity = ens_stats[4]
                entry = f'{encoded_str}, {fault_type}, {fault_amount}, {accuracy}, {resilience}, {diversity}, {elapsed_time}'
                f.write(entry)
                f.write("\n")


def retrieve_golden_models(dataset, ground_labels):
    df = pd.DataFrame(ground_labels, columns=["ground"])
    testset_size = len(ground_labels)
    golden_models = {}

    for idx, modelname in enumerate(model_list):
        modelfilepath = "./injection/" +  dataset + "/golden/arch-" + modelname + "-golden-0"

        with open(modelfilepath, "r") as f:
            ground_labels = json.load(f)
            df[modelname] = ground_labels

            corr = df[df[modelname]==df["ground"]].shape[0]
            acc = corr / testset_size

            golden_models[idx] = acc

    sorted_golden_models = sorted(golden_models, key=golden_models.get, reverse=True)
    return golden_models, sorted_golden_models


def initialize_population(dataset, ground_labels, total_models, ens_size, num_div_ops, k_candidates):
    golden_models, sorted_golden_models = retrieve_golden_models(dataset, ground_labels)
    best_individual_models = sorted_golden_models[:5]
    initial_ensembles = []
    N = total_models * num_div_ops

    for best_model_idx in best_individual_models[:total_models]:
        best_individual_models.append(best_model_idx + total_models)
        best_individual_models.append(best_model_idx + 2*total_models)

    ens_size_arr = [3, 5, 7]

    for k in range(k_candidates):
        for ens_size_k in ens_size_arr:

            if ens_size_k > ens_size:
                break

            selected_indices = random.sample(best_individual_models, ens_size_k)
            encoded_arr = [0] * N
            for idx in selected_indices:
                encoded_arr[idx] = 1

            initial_ensembles.append(encoded_arr)

    return initial_ensembles


def print_best_ensemble(dataset, fault_type, num_fault_amts, best_ensembles, elapsed_time, symmetric):
    elapsed_time = str(int(elapsed_time))
    if symmetric:
        desemble_logfilename = "./output/" + dataset + "_" + fault_type + "_dsemble_best_" + identifier
    else:
        desemble_logfilename = "./output/" + dataset + "_" + fault_type + "_asymmetric_dsemble_best_" + identifier

    min_ad = 1
    min_ad_ens = ""

    for encoded_str, ens_stat_list in best_ensembles.items():
        avg_ad = 0
        for ens_stat in ens_stat_list[1:]:
            avg_ad += ens_stat[3]
        avg_ad /= num_fault_amts
        if avg_ad < min_ad:
            min_ad = avg_ad
            min_ad_ens = encoded_str

    with open(desemble_logfilename, "a") as f:
        ens_stat_list = best_ensembles[min_ad_ens]
        for i in range(1, num_fault_amts+1):
            ens_stats = ens_stat_list[i]
            fault_amount = ens_stats[1]
            accuracy = ens_stats[2]
            resilience = ens_stats[3]
            diversity = ens_stats[4]
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

    ens_size = args.ens_size

    max_iterations = args.max_iterations
    crossover_prob = args.crossover_prob
    mutation_prob = args.mutation_prob

    # Hyperparameters that are fixed for now
    N = total_models = 7
    num_div_ops = 3

    k_candidates = 10
    top_k_candidates = 5

    fault_amt_arr = get_fault_amt_range(fault_amount_min, fault_amount_max)

    num_fault_amts = len(fault_amt_arr)

    ground_labels, golden_labels = read_ground_golden_labels(dataset)

    ens_map = {} #{"encoded_str": [fitness, N]}
    ens_fields = {} # Global map -> {"encoded_str": [total_resilience, [fault_type, fault_amt, accuracy, resilience, diversity], [fault_type, fault_amt, accuracy, resilience, diversity]]}
    population = initialize_population(dataset, ground_labels, total_models, ens_size, num_div_ops, k_candidates)

    start = time.time()

    for iteration_idx in range(max_iterations):

        next_population = []

        for candidate in population:
            for fault_amt in fault_amt_arr:
                final_fault = fault_type + "-" + fault_amt
                encoded_arr = candidate
                encoded_str = ''.join(map(str, encoded_arr))
                fitness, accuracy, resilience, diversity = eval_fitness(encoded_arr, dataset, ground_labels, golden_labels, final_fault, symmetric, num_epochs, batch_size, partition, alpha_zero, acc_metric)

                if encoded_str in ens_map:
                    ens_val = ens_map[encoded_str]
                    fitval = ens_val[0]
                    N = ens_val[1]
                    fitval = (fitval * N + fitness)/(N+1)
                    fitness = fitval
                    ens_map[encoded_str] = [fitval, N+1]
                else:
                    ens_map[encoded_str] = [fitness, 1]

                if encoded_str not in ens_fields:
                    ens_fields[encoded_str] = [fitness]
                else:
                    ens_fields[encoded_str][0] += fitness

                if len(ens_fields[encoded_str]) < num_fault_amts + 1:
                    ens_fields[encoded_str].append([fault_type, fault_amt, accuracy, resilience, diversity])

        best_candidates_encoding_str = list(dict(sorted(ens_fields.items(), key=lambda item: item[1], reverse=True)).keys())[0:top_k_candidates]
        print("\nbest_candidates_encoding_str:   ", dict(sorted(ens_fields.items(), key=lambda item: item[1], reverse=True)))
        best_ensembles = {estr: ens_fields[estr] for estr in best_candidates_encoding_str}
        print("\nbest_ensembles:   ", best_ensembles)

        end = time.time()
        elapsed_time = end - start

        log(dataset, fault_type, num_fault_amts, best_ensembles, elapsed_time, symmetric)
        print_best_ensemble(dataset, fault_type, num_fault_amts, best_ensembles, elapsed_time, symmetric)

        best_candidates_encoding_str = best_candidates_encoding_str[:int(top_k_candidates)]

        for candidate in best_candidates_encoding_str:
            encoded_arr = decode_str(candidate)
            next_population.append(encoded_arr)

        num_population = len(next_population)

        for i in range(num_population-1):
            encoded_arrA = next_population[i]
            encoded_arrB = next_population[i+1]

            print("\nEntering crossover...")

            encoded_arrA, encoded_arrB = crossover(encoded_arrA, encoded_arrB, total_models, ens_size, num_div_ops, crossover_prob)
            next_population.append(encoded_arrA)
            next_population.append(encoded_arrB)

        for i in range(num_population):
            encoded_arr = next_population[i]
            encoded_arr = mutation(encoded_arr, total_models, ens_size, num_div_ops, mutation_prob)
            next_population.append(encoded_arr)

        print("\nMoving to next population...")
        print(next_population)

        population = next_population


if __name__ == "__main__":
    main()


