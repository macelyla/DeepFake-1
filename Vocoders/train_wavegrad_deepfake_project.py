import os
import sys
from trainer import Trainer, TrainerArgs
# Path where the TTS modules are located
tts_module_path = '/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/TTS'
# Append this path to sys.path
sys.path.append(tts_module_path)
from TTS.utils.audio import AudioProcessor
from TTS.config.shared_configs import BaseAudioConfig
from TTS.vocoder.configs import WavegradConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.wavegrad import Wavegrad
import subprocess

output_path = "./logs_wave_deepfake_project"


audio_config = BaseAudioConfig(
    num_mels=80,
    fft_size=2048,
    sample_rate=24000,
    win_length=1024,
    hop_length=256,
    frame_length_ms=None,
    frame_shift_ms=None,
    preemphasis=0.0,
    min_level_db=-100,
    ref_level_db=20,
    power=1.0,
    griffin_lim_iters=60,
    log_func="np.log10",
    stft_pad_mode="reflect",
    signal_norm=True,
    symmetric_norm=True,
    max_norm=4.0,
    clip_norm=True,
    mel_fmin=0.0,
    mel_fmax=12000.0,
    spec_gain=20.0,
    do_trim_silence=False,
    trim_db=60
)

config = WavegradConfig(
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=6144,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=50,
    print_step=50,
    print_eval=True,
    mixed_precision=False,
    data_path="/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/DeepFake/Aine_tacotron_spanish/vocoders/deep_fake_project/df_datset/wav_files",
    output_path=output_path,
    save_step=3000,
    save_checkpoints=True,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model= Wavegrad(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
