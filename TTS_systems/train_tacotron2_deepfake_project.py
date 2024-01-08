import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.models.vits import CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_tacotron2_deep_fake_project")

dataset_config = BaseDatasetConfig(
    formatter="cml_tts", 
    meta_file_train="portuguese_df_trainlist.csv", 
    path="/run/media/fred/FRED1TB/Datasets/cml_tts_dataset_portuguese_v0.1/"
)

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

config = Tacotron2Config(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=10,
    #r=6,
    #gradual_training=[[0, 6, 48], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=False,
    epochs=1000,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    characters=CharactersConfig(
        characters_class="TTS.tts.models.vits.VitsCharacters",
        pad="_",
        eos="&",
        bos="*",
        blank=None,
        characters = "abcdefghijklmnopqrstuvwxyz£ßàáâãäçèéêëìíîïñòóôõöùúûüœ",
        punctuations="—¡!'(),-.·:;?¿ ",
        phonemes=None,
        is_unique=True,
        is_sorted=True,
    ),
    print_step=25,
    print_eval=True,
    save_step=10000,
    save_n_checkpoints=2,
    save_checkpoints=True,
    save_all_best=False,
    mixed_precision=False,
    use_speaker_embedding=False,
    use_d_vector_file=True,
    d_vector_file="speaker_embeddings_deepfake_project/embeddings_cml_portuguese.json",
    d_vector_dim=192,
    min_text_len=5,
    max_text_len=500,
    min_audio_len=24000 * 3,
    max_audio_len=24000 * 20,
    output_path=output_path,
    datasets=[dataset_config],
    lr=1e-4,
)

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

# init speaker manager for multi-speaker training
# it mainly handles speaker-id to speaker-name for the model and the data-loader
speaker_manager = SpeakerManager.init_from_config(config)
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

