import torch
import os
import pandas as pd
from trainer import Trainer, TrainerArgs
import gruut
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CapacitronVAEConfig
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.tacotron2 import Tacotron2

'''CHANGED FORMATTER TO BE USING THE FIRST 100 ROWS OF DATA'''

print(torch.cuda.is_available())
USE_CUDA = torch.cuda.is_available()

file_output = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1/tmp"

output_path = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Create a BaseDatasetConfig object
dataset_config = BaseDatasetConfig(formatter="custom_formatter2", meta_file_train="train.csv", path=os.path.join(output_path))


# Load the dataset using your custom formatter

audio_config = BaseAudioConfig(
    sample_rate=24000,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=11025,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

# Using the standard Capacitron config
capacitron_config = CapacitronVAEConfig(capacitron_VAE_loss_alpha=1.0, capacitron_capacity=50)

config = Tacotron2Config(
    run_name="Capacitron-Tacotron2",
    audio=audio_config,
    capacitron_vae=capacitron_config,
    use_capacitron_vae=True,
    batch_size=32,  # Tune this to your gpu
    max_audio_len=40 * 24000,  # Tune this to your gpu
    min_audio_len=2 * 24000,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=6,
    ga_alpha=0.0,
    r=2,
    optimizer="CapacitronOptimizer",
    optimizer_params={"RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6}, "SGD": {"lr": 1e-5, "momentum": 0.9}},
    attention_type="dynamic_convolution",
    grad_clip=0.0,  # Important! We overwrite the standard grad_clip with capacitron_grad_clip
    double_decoder_consistency=False,
    epochs=25,
    use_phonemes=True,
    phoneme_language="es",
    phonemizer="gruut",
    phoneme_cache_path=os.path.join(file_output, "phoneme_cache3"),
    stopnet_pos_weight=15,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    seq_len_norm=False,
    output_path=file_output,
    datasets=[dataset_config],
    use_speaker_embedding=True,
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
    # Need to experiment with these below for capacitron
    loss_masking=False,
    decoder_loss_alpha=1.0,
    postnet_loss_alpha=1.0,
    postnet_diff_spec_alpha=0.0,
    decoder_diff_spec_alpha=0.0,
    decoder_ssim_alpha=0.0,
    postnet_ssim_alpha=0.0,
)

ap = AudioProcessor(**config.audio.to_dict())

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.num_speakers = speaker_manager.num_speakers

model = Tacotron2(config, ap, tokenizer, speaker_manager=speaker_manager)

trainer = Trainer(
    TrainerArgs(),
    config,
    file_output,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)

trainer.fit()