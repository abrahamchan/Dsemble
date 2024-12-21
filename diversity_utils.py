#! /usr/bin/python3

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical

from keras import layers, models

from arch.resnet import get_resnet_by_n
from arch.vgg import get_vgg16, get_vgg11
from arch.convnet import get_convnet, get_deconvnet
from arch.mobilenet import get_mobilenet

import os.path

from src import tfi, config

import numpy as np
import random
import math
import json
import pandas as pd


def get_model_by_name(model_type, input_shape):
    if model_type == "ResNet50":
        model = get_resnet_by_n(9, input_shape, classes=num_classes)
    elif model_type == "ResNet18":
        model = get_resnet_by_n(3, input_shape, classes=num_classes)
    elif model_type == "VGG11":
        model = get_vgg11(input_shape, classes=num_classes)
    elif model_type == "ConvNet":
        model = get_convnet(input_shape, classes=num_classes)
    elif model_type == "DeconvNet":
        model = get_deconvnet(input_shape, classes=num_classes)
    elif model_type == "MobileNet":
        model = get_mobilenet(input_shape, classes=num_classes)
    else:
        model = get_vgg16(input_shape, classes=num_classes)


    model.compile(loss='sparse_categorical_crossentropy',
                 optimizer='adam',
                 metrics=['acc'])

    return model


def process_mnist_images(images):
    images = images[:, :, :, np.newaxis]
    images = np.pad(images, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant")
    return images


def compute_dataset_noise_percent(total_per_class, diagonals):
    total_elements = np.sum(total_per_class)
    sum_diagonals = np.sum(diagonals)
    avg_dataset_noise_percent = float((total_elements - sum_diagonals) / total_elements)
    noise_per_class = total_per_class - diagonals
    return noise_per_class, avg_dataset_noise_percent


def generate_noise_matrix(noise_matrix_file, fault_amt, threshold = 0.01):
    np.seterr(divide='ignore', invalid='ignore')

    transition_df = pd.read_csv(noise_matrix_file, index_col=0)
    num_classes = transition_df.shape[0]

    target_dataset_noise_percent = fault_amt / 100.

    transition_matrix = transition_df.to_numpy()
    total_per_class = transition_matrix.sum(axis=0)

    diagonals = np.diagonal(transition_matrix)
    noise_per_class, avg_dataset_noise_percent = compute_dataset_noise_percent(total_per_class, diagonals)

    print("Initial Noise: ", avg_dataset_noise_percent)

    while True:
        noise_scaling_factor = target_dataset_noise_percent / avg_dataset_noise_percent
        max_noise_limit = (noise_per_class * noise_scaling_factor).clip(None,total_per_class)
        noise_scaling_factor_per_class = np.nan_to_num(max_noise_limit / noise_per_class)

        for row in range(num_classes):
            for column in range(num_classes):
                if row != column:
                    transition_matrix[row][column] = round(transition_matrix[row][column] * noise_scaling_factor_per_class[column])

        new_diagonals = (total_per_class - noise_per_class * noise_scaling_factor_per_class).clip(0)
        np.fill_diagonal(transition_matrix, new_diagonals)

        total_per_class_new = transition_matrix.sum(axis=0)
        total_per_class_diff = total_per_class_new - total_per_class

        for column in range(num_classes):
            count_diff = total_per_class_diff[row]
            if total_per_class_diff[column] > 0:
                row = 0
                while count_diff > 0 and row < num_classes:
                    if transition_matrix[row][column] > 0:
                        transition_matrix[row][column] -= 1
                        count_diff -= 1
                    row += 1
            elif total_per_class_diff[column] < 0:
                transition_matrix[column][column] += abs(count_diff)

        noise_per_class, avg_dataset_noise_percent = compute_dataset_noise_percent(total_per_class, new_diagonals)

        if abs(target_dataset_noise_percent - avg_dataset_noise_percent) <= threshold:
            print("Resulting Noise: ", avg_dataset_noise_percent)
            print(transition_matrix)
            return transition_matrix


def inject_asymmetric(y_train, transition_matrix):
    num_classes = transition_matrix.shape[0]
    y_train_before = y_train.copy()

    for category in range(num_classes):
        indices_of_category = np.flatnonzero(y_train_before==category)
        noisy_arr = transition_matrix[:, category]
        repeat_arr = np.repeat(range(num_classes), noisy_arr)
        np.random.shuffle(repeat_arr)
        np.put(y_train, indices_of_category, repeat_arr)

    fault_rate = float(np.sum(y_train_before != y_train) / y_train.shape[0])
    print("Actual Fault Rate: ", fault_rate)
    return y_train


def perform_data_fi_asymmetric(y_train, dataset, final_fault):
    noise_matrix_file = "./noise_matrix/" + dataset + ".csv"
    fault_type, fault_amt = tuple(final_fault.split("-"))
    if fault_type == "label_err":
        transition_matrix = generate_noise_matrix(noise_matrix_file, int(fault_amt))
        y_train = inject_asymmetric(y_train, transition_matrix)
    return y_train


def load_training_data(dataset, final_fault, symmetric=True, is_subset=False, partition=63, overlap=0):
    global num_classes

    if dataset == "cifar10":
        # load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset == "gtsrb":
        (x_train, y_train), (x_test, y_test) = load_gtsrb()
        num_classes = 43
    elif dataset == "pneumonia":
        (x_train, y_train), (x_test, y_test) = load_pneumonia()
        num_classes = 2
    elif dataset == "mnist":
        # load the MNIST data.
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10
        x_train = process_mnist_images(x_train)
        x_test = process_mnist_images(x_test)
    else:
        # load the CIFAR10 data.
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()


    # prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), "faulty_dataset/" + dataset)

    if symmetric:
        fault_npy = final_fault + ".npy"
    else:
        fault_npy = final_fault + "-asymmetric.npy"


    # Reuse fault injected dataset if found
    reuse_dataset = False
    if final_fault != "golden":

        imagefile = os.path.join(save_dir, "image-" + fault_npy)
        labelfile = os.path.join(save_dir, "labels-" + fault_npy)

        if os.path.exists(imagefile) and os.path.exists(labelfile):
            x_train = np.load(imagefile)
            y_train = np.load(labelfile)
            reuse_dataset = True
            print("Existing fault files read")

    if final_fault != "golden" and not reuse_dataset:
        if symmetric:
            x_train, y_train = perform_data_fi(x_train, y_train, final_fault)
            print("New fault injection performed")
        else:
            y_train = perform_data_fi_asymmetric(y_train, dataset, final_fault)
            print("New asymmetric fault injection performed")

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        imagefile = os.path.join(save_dir, "image-" + fault_npy)
        np.save(imagefile, x_train)

        labelfile = os.path.join(save_dir, "labels-" + fault_npy)
        np.save(labelfile, y_train)
    elif not final_fault:
        print("Golden run")


    if is_subset:
        x_train, y_train = random_sample(x_train, y_train, partition)
        print("New shape y: ", y_train.shape[0])

    return setup_data(x_train, y_train, x_test, y_test)


def perform_data_fi(
    train_images,
    train_labels,
    final_fault):

    conf_file = "./confFiles/" + final_fault + ".yaml"

    if (final_fault.startswith("label_err")):
        tf_res = tfi.inject(y_test=train_labels, confFile=conf_file)
        train_labels = tf_res
    else:
        train_images, train_labels = tfi.inject(x_test=train_images,y_test=train_labels, confFile=conf_file)

    print("\n\nLength of labels:  " + str(len(train_labels)) + "\n\n")

    return train_images, train_labels


def setup_data(x_train, y_train, x_test, y_test):
    # normalize data.
    x_train = np.array(x_train).astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train, y_train), (x_test, y_test)

def setup_two_data(x_train_A, y_train_A, x_train_B, y_train_B, x_test, y_test):
    # normalize data.
    x_train_A = np.array(x_train_A).astype('float32') / 255
    x_train_B = np.array(x_train_B).astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return (x_train_A, y_train_A), (x_train_B, y_train_B), (x_test, y_test)


def load_gtsrb():
    root_dir = "./data/GTSRB/"
    train_root_dir = root_dir + "Final_Training/Resized_Images/"
    test_root_dir = root_dir + "Final_Test/"

    import glob
    from skimage import io
    import pandas as pd

    def __read_train_data(train_root_dir):
        imgs = []
        labels = []

        all_img_paths = glob.glob(os.path.join(train_root_dir, '*/*.ppm'))
        np.random.shuffle(all_img_paths)
        for img_path in all_img_paths:
            img = io.imread(img_path)
            label = __get_class(img_path)
            imgs.append(img)
            labels.append(label)

        train_images = np.array(imgs, dtype='float32')
        train_labels = np.array(labels)
        return train_images, train_labels

    def __read_test_data(test_root_dir):
        test = pd.read_csv(test_root_dir + "Labels/GT-final_test.csv", sep=';')

        # Load test dataset
        x_test = []
        y_test = []
        for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
            img_path = os.path.join(test_root_dir + "Resized_Images/", file_name)
            img = io.imread(img_path)

            x_test.append(img)
            y_test.append(class_id)

        test_images = np.array(x_test, dtype='float32')
        test_labels = np.array(y_test)
        return test_images, test_labels

    def __get_class(img_path):
            return int(img_path.split('/')[-2])

    train_images, train_labels = __read_train_data(train_root_dir)
    test_images, test_labels = __read_test_data(test_root_dir)

    return (train_images, train_labels), (test_images, test_labels)


def load_pneumonia():
    pneumonia_root = ""
    train_root_dir = pneumonia_root + "train"
    test_root_dir = pneumonia_root + "test"

    import glob
    from skimage import io

    def __read_train_data(root_dir):
        pixel_size = 128
        target_size = (pixel_size, pixel_size)
        batch_size = 64
        train_datagen = ImageDataGenerator(
                                           shear_range = 0.2,
                                           zoom_range = 0.2,
                                           horizontal_flip = True)
        training_set = train_datagen.flow_from_directory(root_dir,
                                                 target_size = target_size,
                                                 batch_size = batch_size,
                                                 class_mode = 'sparse',
                                                 color_mode='grayscale',
                                                 shuffle=False)
        return __get_data_label(training_set)

    def __read_test_data(root_dir):
        imgs = []
        labs = []

        all_img_paths = sorted(glob.glob(os.path.join(root_dir, '*/*.jpeg')))
        for img_path in all_img_paths:
            img = io.imread(img_path)
            label = __get_class(img_path)
            imgs.append(img)
            labs.append(label)

        images = np.array(imgs, dtype='float32')
        labels = np.array(labs, dtype='int')
        images = np.expand_dims(images, axis=3)
        return images, labels

    def __get_class(img_path):
        img_class = img_path.split('/')[-2]
        return 0 if img_class == "NORMAL" else 1

    def __get_data_label(training_generator):
        batch_index = 0
        while batch_index <= training_generator.batch_index:
            data = training_generator.next()
            if batch_index == 0:
                data_list = data[0]
                label_list = data[1]
            else:
                data_list = np.concatenate((data_list, data[0]), axis=0)
                label_list = np.concatenate((label_list, data[1]))
            batch_index = batch_index + 1
            label_list = label_list.astype(int)
        return data_list, label_list

    train_images, train_labels = __read_train_data(train_root_dir)
    test_images, test_labels = __read_test_data(test_root_dir)

    return (train_images, train_labels), (test_images, test_labels)


def get_trained_model(dataset, model_name, x_train, y_train, num_epochs, batch_size):
    input_shape = x_train.shape[1:]
    model = get_model_by_name(model_name, input_shape)

    model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size)
    print(model_name + " partially trained")

    model.compile(loss='sparse_categorical_crossentropy',
         optimizer='adam',
         metrics=['acc'])

    #model.save_weights(trained_model_file)
    print(model_name + " trained")

    return model


def random_sample(x_train, y_train, subset_percent = 63):
    num = x_train.shape[0]
    sz = (subset_percent * num) / 100
    sz = math.floor(sz)
    ind = random.sample(range(num), k=sz)
    x_train_, y_train_ = tf.gather(x_train, ind), tf.gather(y_train, ind)
    return (x_train_, y_train_)


def write_preds_file(model, x_test, div_op, dataset_name, model_name, final_fault, symmetric, identifier):
    preds_test = model.predict(x_test)
    predictions = np.argmax(preds_test, axis=1)
    pred_list = predictions.tolist()

    if final_fault == "golden":
        fault_type = golden
    else:
        fault_type = final_fault.split('-')[0]

    if symmetric:
        pred_file_name = "./injection/" +  dataset_name + "/" + fault_type + "/" + div_op + "-" + model_name + "-" + final_fault + "-" + str(identifier)
    else:
        pred_file_name = "./injection/" +  dataset_name + "/" + fault_type + "/" + div_op + "-" + model_name + "-" + final_fault + "-asymmetric-" + str(identifier)

    with open(pred_file_name, "w") as w_file:
        json.dump(pred_list, w_file)


def get_majority_vote(df):
    temp_df = df
    temp_df2 = temp_df.mode(axis='columns', numeric_only=True)
    temp_df2["Mode"] = temp_df2.loc[:,0].astype(int)
    return temp_df2


def get_ground_truth(dataset):
    with open("./groundtruth/" + dataset, "r") as r_file:
        return json.load(r_file)


def write_ensemble_decision(encoded_arr, pred_filenames, dataset, final_fault, symmetric):
    pred_pd_list = []
    for pred_filename in pred_filenames:
        df = pd.read_json(pred_filename)
        pred_pd_list.append(df)

    modedf = pd.concat(pred_pd_list, axis=1, ignore_index=True)

    resultdf = modedf.copy()
    modedf = get_majority_vote(modedf)
    modelist = modedf["Mode"].tolist()
    resultdf["ens"] = modedf["Mode"]

    correctdf = resultdf.copy()
    correctdf.insert(0, "ground", pd.DataFrame(get_ground_truth(dataset)))

    for column in correctdf.columns[1:]:
        correctdf[column] = np.where(correctdf[column]==correctdf["ground"], 1, 0)

    correctdf = correctdf.drop(columns=["ground"])

    if final_fault == "golden":
        fault_type = golden
    else:
        fault_type = final_fault.split('-')[0]

    encoded_str = ''.join(map(str, encoded_arr))

    if symmetric:
        fname_prefix = "./injection/" +  dataset + "/" + fault_type + "/" + final_fault + "-" + encoded_str
    else:
        fname_prefix = "./injection/" +  dataset + "/" + fault_type + "/" + final_fault + "-asymmetric-" + encoded_str

    resultdf.to_csv(fname_prefix + "-pred.csv", index=False)
    correctdf.to_csv(fname_prefix + "-correct.csv", index=False)

