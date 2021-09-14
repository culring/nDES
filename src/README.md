# Neural Differential Evolution Strategy (nDES)

## Installation

### Prerequisites
* Cuda toolkit v10.2 with a corresponding driver
* `nvcc` in your $PATH

### Python libraries
Create a new [`conda`](https://conda.io/) environment and install libraries from environment.yml:
```shell
conda env create -f environment.yml
``` 
Activate the environment and install additional packages:
```shell
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```
Add the nDES package to the PYTHONPATH environment variable:
```shell
export PYTHONPATH=${PYTHONPATH}:{sciezka_do_ndes}
```

### Install CUDA-based extensions
```shell
cd gpu_utils && python setup.py install
```
You can test these extensions by running [`pytest`](https://pytest.org/) in the `gpu_utils` directory after
you've installed the extensions.

## Usage
### Fashion MNIST Experiment
To run an experiment issue `python fashion_mnist_experiment.py`. To configure it you
need to change global variables on the top of the file.

### RNN Toy Problems
These experiments take a cuda device number as an argument, as well as sequence length.
Example usage: `python rnn_addition_experiment.py -cuda 0 20`.
