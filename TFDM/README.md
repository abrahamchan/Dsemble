# TensorFlow Data Mutator (TF-DM)

TensorFlow Data Mutator (TF-DM) is a framework for injecting different data faults into ML applications written using the TensorFlow 2 framework.

We support five types of data faults: data removal, data repetition, data shuffle, noise addition and label errors. We also support four types of noise addition and targeted misclassification in label errors.

Each fault is specified with the fault type and an integer parameter, called the amount, which represents the amount of perturbation. This latter parameter is unique for each fault type. Both can be configured for FI runs through YAML.
