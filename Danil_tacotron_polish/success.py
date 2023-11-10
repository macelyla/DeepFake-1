import torch
import os
import pandas as pd
from trainer import Trainer, TrainerArgs
import gruut
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.tacotron2 import Tacotron2


print(torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()

file_output = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1/tmp"

output_path = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create a BaseDatasetConfig object
dataset_config = BaseDatasetConfig(formatter="custom_formatter2", meta_file_train="train.csv", path=os.path.join(output_path))



audio_config = BaseAudioConfig(
    sample_rate=24000,                        # Resample to 22050 Hz. It slows down training. Use `TTS/bin/resample.py` to pre-resample and set this False for faster training.
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=11025,
    ref_level_db=20,
    spec_gain=1.0,
    log_func="np.log",
    preemphasis=0.0,

)

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    max_decoder_steps=100000,
    batch_size=32,  # Tune this to your gpu
    max_audio_len=40 * 24000,  # Tune this to your gpu
    min_audio_len=2 * 24000,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=4,
    run_eval=True,
    ga_alpha=0.0,
    test_delay_epochs=-1,
    r=6,
    gradual_training=[[0, 6, 64], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=True,
    #r=2,
    #gradual_training=[[0, 6, 48], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    #double_decoder_consistency=False,
    epochs=100,
    phonemizer="gruut",
    use_phonemes=True,
    phoneme_language="es",
    phoneme_cache_path=os.path.join(file_output, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    #115,250
    min_text_len=115,
    max_text_len=250,
    min_audio_len=22050 * 0,
    max_audio_len=22050 * 33,
    output_path=file_output,
    datasets=[dataset_config],
    use_speaker_embedding=True,  # set this to enable multi-sepeaker training
    decoder_ssim_alpha=0.0,  # disable ssim losses that causes NaN for some runs.
    postnet_ssim_alpha=0.0,
    postnet_diff_spec_alpha=0.0,
    decoder_diff_spec_alpha=0.0,
    attention_norm="softmax",
    optimizer="Adam",
    lr_scheduler=None,
    lr=3e-5,
    test_sentences=[
        "En un lugar de la Mancha, de cuyo nombre quiero acordarme, no hace mucho que vivía un hidalgo.",
        "Aprender a dudar es aprender a pensar",
        "Se necesitan dos años para aprender a hablar y sesenta para aprender a callar.",
        "Mi perro tiene dos colas y tres ojos",
        "Mi madre me dio un regalo por mi cumpleaños y me gustó",
    ]
)

## INITIALIZE THE AUDIO PROCESSOR
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


print("Number of training samples:", len(train_samples))
print("Number of evaluation samples:", len(eval_samples))


# Handle Empty Dataset
if not train_samples or not eval_samples:
    print("Error: Training or evaluation samples are empty.")

# init speaker manager for multi-speaker training
# it mainly handles speaker-id to speaker-name for the model and the data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")

# init model
model = Tacotron2(config, ap, tokenizer, speaker_manager)


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
model = model.to(device)  # Where `device` is either 'cuda' or 'cuda:0' if you have multiple GPUs.

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
# AND... 3,2,1... 🚀
trainer.fit()