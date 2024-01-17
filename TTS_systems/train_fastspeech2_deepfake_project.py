import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fastspeech2_config import Fastspeech2Config
from TTS.tts.models.vits import CharactersConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from TTS.utils.manage import ModelManager

output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs_fastspeech2_deep_fake_project")


# init configs
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

config = Fastspeech2Config(
    audio=audio_config,
    batch_size=4,
    eval_batch_size=4,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    compute_energy=True,
    energy_cache_path=os.path.join(output_path, "energy_cache"),
    run_eval=True,
    test_delay_epochs=-1,
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
    precompute_num_workers=4,
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
    eval_split_size=0.1
)

# compute alignments
if not config.model_args.use_aligner:
    manager = ModelManager()
    # TODO: Use your trained tacotron version
    # model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/tacotron2-DCA")
    os.system(
        f"python TTS/bin/compute_attention_masks.py --model_path {model_path} --config_path {config_path} --dataset ljspeech --dataset_metafile metadata.csv --data_path ./recipes/ljspeech/LJSpeech-1.1/  --use_cuda true"
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
#speaker_manager = SpeakerManager.init_from_config(config)
speaker_manager = SpeakerManager(d_vectors_file_path="speaker_embeddings_deepfake_project/embeddings_cml_portuguese.json")
#speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")


# init the model
model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()