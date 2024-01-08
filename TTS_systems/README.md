# Coqui.ai 🐸

## Installation

Coqui.ai Text-to-Speech (TTS) is tested on Ubuntu 18.04 with **python >= 3.9, < 3.12**.

1. Create an Anaconda environment:

    ```bash
    conda create -n coqui-tts python=3.10 pip
    conda activate coqui-tts
    ```

2. Install PyTorch (Check [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)):

    ```bash
    pip3 install torch torchvision torchaudio
    ```

3. Clone Coqui.ai TTS and install it locally:

    ```bash
    git clone https://github.com/coqui-ai/TTS
    cd TTS
    pip install -e .[all]
    ```

## Docker Image

You can also try Coqui.ai TTS without installing by using the Docker image. Run the following commands:

```bash
cd TTS
sudo docker build -t coqui-tts ./
sudo docker run --net='host' --rm --shm-size=2g --runtime=nvidia -v /home/source:/root -w /root -it coqui-tts bash
```

## Dataset Preparation

Download and unzip the datasets:

```bash
mkdir DATASETS
cd DATASETS
wget -c https://www.openslr.org/resources/146/cml_tts_dataset_dutch_v0.1.tar.bz
wget -c https://www.openslr.org/resources/146/cml_tts_dataset_french_v0.1.tar.bz
wget -c https://www.openslr.org/resources/146/cml_tts_dataset_german_v0.1.tar.bz
wget -c https://www.openslr.org/resources/146/cml_tts_dataset_spanish_v0.1.tar.bz

tar -jxvf cml_tts_dataset_dutch_v0.1.tar.bz
tar -jxvf cml_tts_dataset_french_v0.1.tar.bz
tar -jxvf cml_tts_dataset_german_v0.1.tar.bz
tar -jxvf cml_tts_dataset_spanish_v0.1.tar.bz
```

## Training Acoustic Models

To train the models, copy the training files to the repository root directory https://github.com/coqui-ai/TTS:

```bash
cp train_tacotron2_deepfake_project.py TTS/
cp train_fastspeech2_deepfake_project.py TTS/
```

Run the training scripts. For example:

```bash
python train_tacotron2_deepfake_project.py
```

For multi-GPU training:

```bash
python3 -m trainer.distribute --gpus "0,1,2" --script train_tacotron2_deepfake_project.py \
```
