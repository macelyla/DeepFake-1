import os
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.tacotron2 import Tacotron2
import torch
import pandas as pd


# Check GPU availability
print(torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()

output_path = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1"

file_output= "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1/tmp"


if not os.path.exists(output_path):
    os.makedirs(output_path)
    

# Create a BaseDatasetConfig object
dataset_config = BaseDatasetConfig(formatter="custom_formatter", meta_file_train="train.csv", path=os.path.join(output_path))


# Load the dataset using your custom formatter

audio_config = BaseAudioConfig(
    sample_rate=24000,
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

# Configure your Tacotron2 model
config = Tacotron2Config(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,
    use_phonemes=True,
    phonemizer="gruut",
    phoneme_language="es",
    phoneme_cache_path=os.path.join(file_output, "phoneme_cache2"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    seq_len_norm=True,
    output_path=file_output,
    datasets=[dataset_config],
    use_speaker_embedding=True,
    min_text_len=0,
    max_text_len=500,
    min_audio_len=100000,  # Adjust this value based on your requirements
    max_audio_len=400000,  # Adjust this value based on your requirements
    lr=1e-3,
    lr_scheduler="StepwiseGradualLR",
    lr_scheduler_params={
        "gradual_learning_rates": [
            [0, 1e-3],
            [2e4, 5e-4],
            [4e5, 3e-4],
            [6e4, 1e-4],
            [8e4, 5e-5],
        ]
    },
    scheduler_after_epoch=False,  # scheduler doesn't work without this flag
    
)

config.audio.fft_size = 2048
config.audio.win_length = 1200
config.audio.hop_length = 256
config.audio.num_mels = 80

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)


ap = AudioProcessor.init_from_config(audio_config)

# Initialize the TTSTokenizer
tokenizer, config = TTSTokenizer.init_from_config(config)

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

# Create the Tacotron2 model
model = Tacotron2(config, ap, tokenizer, speaker_manager=speaker_manager)

# Train the model on a single GPU (no need for DataParallel)
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)

# Train the model
trainer.fit()


