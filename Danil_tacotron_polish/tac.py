import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, CapacitronVAEConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.speakers import SpeakerManager
import torch
torch.cuda.empty_cache()



file_output = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1/tmp"

output_path = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# Using LJSpeech like dataset processing for the blizzard dataset
dataset_config = BaseDatasetConfig(formatter="custom_formatter2", meta_file_train="train.csv", path=os.path.join(output_path))


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
    batch_size=16,  # Tune this to your gpu
    max_audio_len=36 * 10000,  # Tune this to your gpu
    min_audio_len=23 * 10000,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=9,
    ga_alpha=0.0,
    r=2,
    optimizer="CapacitronOptimizer",
    optimizer_params={"RAdam": {"betas": [0.9, 0.998], "weight_decay": 1e-6}, "SGD": {"lr": 1e-5, "momentum": 0.9}},
    attention_type="dynamic_convolution",
    grad_clip=0.0,  # Important! We overwrite the standard grad_clip with capacitron_grad_clip
    double_decoder_consistency=False,
    epochs=10,
    use_phonemes=True,
    phoneme_language="es",
    phonemizer="gruut",
    phoneme_cache_path=os.path.join(file_output, "phoneme_cache10"),
    stopnet_pos_weight=15,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    seq_len_norm=True,
    use_speaker_embedding=True,
    output_path=file_output,
    datasets=[dataset_config],
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

# init speaker manager for multi-speaker training
# it mainly handles speaker-id to speaker-name for the model and the data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")

# init model
model = Tacotron2(config, ap, tokenizer, speaker_manager)

# INITIALIZE THE TRAINER
# Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
# distributed training, etc.
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

# AND... 3,2,1... 🚀
trainer.fit()


