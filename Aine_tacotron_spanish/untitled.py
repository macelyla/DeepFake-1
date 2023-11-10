import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

import torch
torch.cuda.empty_cache()

print(torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()

file_output = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1/tmp"

output_path = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create a BaseDatasetConfig object
dataset_config = BaseDatasetConfig(formatter="custom_formatter2", meta_file_train="train.csv", path=os.path.join(output_path))

audio_config = BaseAudioConfig(
    sample_rate=24000,
    do_trim_silence=True,
    trim_db=40.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=35,
    preemphasis=0.0,
)

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=64,
    eval_batch_size=32,
    num_loader_workers=12,
    num_eval_loader_workers=12,
    precompute_num_workers=24,
    run_eval=True,
    max_decoder_steps=20000,
    test_delay_epochs=-1,
    r=6,
    gradual_training=[[0, 6, 64], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=True,
    epochs=400,
    use_phonemes=True,
    phonemizer="gruut",                           
    phoneme_language="es",
    phoneme_cache_path=os.path.join(file_output, "phoneme_cache3"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=file_output,
    datasets=[dataset_config],
    test_sentences=[
        "En un lugar de la Mancha, de cuyo nombre quiero acordarme.",
        "Aprender a dudar es aprender a pensar.",
        "Se necesitan dos años para aprender a hablar y sesenta para aprender a callar.",
        "Mi perro tiene dos colas y tres ojos.",
        "Mi madre me dio un regalo por mi cumpleaños y me gustó."
    ],
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# If characters are not defined in the config, default characters are passed to the config
tokenizer, config = TTSTokenizer.init_from_config(config)

# LOAD DATA SAMPLES
# Each sample is a list of ```[text, audio_file_path, speaker_name]```
# You can define your custom sample loader returning the list of samples.
# Or define your custom formatter and pass it to the `load_tts_samples`.
# Check `TTS.tts.datasets.load_tts_samples` for more details.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# INITIALIZE THE MODEL
# Models take a config object and a speaker manager as input
# Config defines the details of the model like the number of layers, the size of the embedding, etc.
# Speaker manager is used by multi-speaker models.
model = Tacotron2(config, ap, tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
