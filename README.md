# ImgVAE

## Setup

Create a Python environment and install the required dependencies:

```
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Note that Python cannot be too new, or else torch cannot be installed. We found success using Python 3.10 in our environment. If your local machine does not support cuda, remove the `--index-url` in the second command and simply install `torch`.

## Training

To start training, run `train.py`. Run `python train.py --help` for a full list of customizable options.

Example command:

```
python train.py --epoch_size 65536 --batch_size 256 --latent_dim 256 --hidden_dim 1024 --num_epochs 150
```

Add the `run_name` and/or `run_timestamp` options to identify the run:

```
python train.py --run_name myrun --run_timestamp
```

Following is a standard training command which provides fair results:

```
python train.py --epoch_size 2048 --batch_size 256 --graph_size 32 --latent_dim 1024 --num_epochs 100 --run_name my-model-name --run_timestamp
```