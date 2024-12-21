# D-semble
Finding ensembles that are resilient, diverse, and accurate against faulty training data.


## Directory Layout

```
.
└── arch                    # Folder containing TensorFlow implementations of model architectures
├── clean.sh                # Script to clean generated files (Rerun training from scratch. Recommended only for stronger GPUs)
└── confFiles               # Contains pre-defined YAML files for experimental configs
└── data                    # Folder to store datasets
│   └── GTSRB               # Directory for the GTSRB dataset
├── decode.py               # Utility tool to decode encoded ensemble strings into human readable format
├── diversity_utils.py      # Helper library for D-semble (**not to be directly invoked**)
├── Dockerfile              # Used to generate Docker image for D-semble
├── dsemble.py              # Main script to invoke D-semble
└── faulty_dataset          # Folder to store fault injected datasets
│   └── gtsrb
└── faulty_labels           # Folder to store Cleanlab generated dataset issues log files
│   └── issue_results_gtsrb.csv
├── fitness.py              # Helper library for D-semble (**not to be directly invoked**)
└── golden                  # Folder containing labels that were correctly classified by golden models
├── greedy_select.py        # Script to invoke greedy ensemble selection
└── groundtruth             # Folder containing ground truth labels for each dataset
└── injection               # Folder containing predictions by models trained on fault injected datasets, organized by fault type
│   └── gtsrb
│       └── golden
│       └── label_err
│       └── remove
│       └── repeat
├── noise_matrix            # Folder to store noise transition matrices
├── noise_transition.py     # Script to generate noise transition matrix from Cleanlab log file
├── partial_clean.sh        # Script to partially clean generated files (Rerun some training. Recommended for weaker GPUs)
├── random_select.py        # Script to invoke random ensemble selection
├── requirements.txt        # List of dependencies
├── snapshot.py             # Helper library for D-semble (**not to be directly invoked**)
├── setup.sh                # Script to populate required empty folders for D-semble
└── output                  # Output folder for D-semble
└── TFDM                    # Folder for training data fault injector tool
```


## Installation

Requirements:

1. Python 3.5+
2. GPU (Note: Performance times can differ with GPU capability, and VRAM availability). If no GPU or only a weak GPU (< 6 GB VRAM) is available, you may still run D-semble with some pretrained files (This is enabled by default for this artifact).

### (Option A) Manual Installation

1. Use pip to install the required dependencies.
```
pip install -r requirements.txt
```

### (Option B) Using Docker Image on Ubuntu 20.04 with NVIDIA CUDA 11.8

Assuming you have Docker (19.03+) installed, you will also need to enable Docker to access your GPU.

1. To run the Docker image with GPU support, you will need to install `nvidia-container-runtime`.

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

2. Modify the Dockerfile so that it matches the CUDA version (default is 11.8) installed on your host machine.

```
sudo docker build -t dsemble:latest .
sudo docker run -t -d --runtime=nvidia --gpus all --name dsemble_container dsemble
sudo docker exec -it dsemble_container /bin/bash
```

### Datasets

1. GTSRB, a preprocessed version of the original, is included in this repository. No further preprocessing is required.
2. CIFAR-10 will be automatically downloaded if the dataset option is set to "cifar10" in D-semble for the first time. No further preprocessing required.
3. Pneumonia can be downloaded [here](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). Images must be resized to 128 x 128 pixels before use.


## Getting Started with a Minimum Working Example

We provide an example of how to use D-semble on the provided GTSRB dataset.
Our objective is to find the most resilient ensemble on GTSRB against mislabelling ranging between 10% to 50%.

1. Run the following command.
```
python dsemble.py --dataset gtsrb --epochs 5 --fault_type label_err --max_iterations 3 --natural
```

Note that D-semble caches results from previous runs to speed up search on subsequent runs. Even with caching, it will still rerun the metaheuristic search.
For convenience, some pre-trained files have already been provided by default.
If you wish to retrain some files (recommended for moderate GPU resources), run `partial_clean.sh` and reinvoke `dsemble.py`.
If you wish to start an entirely new search from scratch and you have sufficient GPU resources, run `clean.sh`, and reinvoke `dsemble.py`.

2. You will see files generated under the `output` folder.
You will see two types of generated files: `<dataset>_<fault_type>_<elapsedtime>` and `<dataset>_<fault_type>_dsemble_best_<run_hash_id>`.

Example of output files generated by D-semble in this exercise.
* `gtsrb_label_err_asymmetric_300`: Top K Resilient Ensemble Candidates for GTSRB against mislabelling. The 300 refers to the elapsed time in seconds.
* `gtsrb_label_err_asymmetric_dsemble_best_6cab009c-76bb-461e-9176-d1e38445b6ea`: The best performing ensembles selected by D-semble

The data in the files are organized as CSVs. The CSV format is as follows:
```
encoded_ensemble, fault_type, fault_amount, accuracy, accuracy_delta, diversity, elapsed_time
```

For example, consider this entry:
```
000000000001000100001, label_err, 30, 0.91, 0.01, 0.34, 300
```
This means that D-semble found an ensemble "000000000001000100001" after 300 seconds. The ensemble has 0.91 balanced accuracy at 30% mislabelling. It also has 0.34 diversity.

3. D-semble reports the encoded numerical ensemble. To decode an ensemble string in human readable form, use the `decode.py` script.
```
python decode.py 000000000001000100001
```

Expected Output:
```
Ensemble combination decoded:  ['Data Div Op on ResNet50', 'Snapshot Div Op on DeconvNet', 'Snapshot Div Op on VGG16']
```
This particular ensemble was constructed with the data diversity operator applied on ResNet50, a snapshot of DeconvNet, and a snapshot of VGG16.



## Reproducing a Result

We provide an example of comparing D-semble with random selection on GTSRB with mislabelling faults (10% to 50%).
We also provide an example of generating a noise transition matrix from the label issues log file, generated by Cleanlab.

1. First, clear the cache directories of any generated log files from the minimum working example.
```
./clean.sh
```

2. Generate the noise transition matrix.

```
python noise_transition.py faulty_labels/issue_results_gtsrb.csv noise_matrix/gtsrb.csv
```

3. Run D-semble.
```
python dsemble.py --dataset gtsrb --epochs 10 --fault_type label_err --max_iterations 5 --natural
```

4. Run random select.
```
python random_select.py --dataset gtsrb --epochs 10 --fault_type label_err --natural
```

5. You will see files generated under the `output` folder.

Example of output files generated in this exercise.
* `gtsrb_label_err_asymmetric_300`: Top K Resilient Ensemble Candidates for GTSRB against mislabelling. The 300 refers to the elapsed time in seconds.
* `gtsrb_label_err_asymmetric_dsemble_best`: The best performing ensembles selected by D-semble
* `gtsrb_label_err_asymmetric_random_select`: A randomly selected ensemble

6. Please see the above section on how to interpret output files. You should see that the balanced accuracy of the ensembles returned by D-semble should be lower than that of the randomly selected ensemble in most runs.


## Additional Configurations to Try (Optional)

We provide examples of how to use D-semble for other experimental configurations - these are not exhaustive.

1. D-semble with other fault types (i.e., remove or repeat).
```
python dsemble.py --dataset gtsrb --epochs 10 --fault_type remove
```

2. D-semble with a custom fault amount range.
```
python dsemble.py --dataset gtsrb --epochs 10 --fault_type label_err --natural --fault_amount_min 10 --fault_amount_max 30
```

3. D-semble to find larger ensembles than 3 (i.e., 5 or 7).
```
python dsemble.py --dataset gtsrb --epochs 10 --fault_type label_err --natural --ens_size 5
```

4. D-semble with accuracy instead of balanced accuracy.
```
python dsemble.py --dataset gtsrb --epochs 10 --fault_type label_err --natural --acc_metric accuracy
```

5. Run greedy select.
```
python greedy_select.py --dataset gtsrb --epochs 10 --fault_type label_err --natural
```


## License

D-semble is released under an Apache 2.0 License.

