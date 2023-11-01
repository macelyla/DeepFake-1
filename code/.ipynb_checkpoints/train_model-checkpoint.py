import os

from trainer import Trainer, TrainerArgs

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.tts.configs.fast_speech_config import FastSpeechConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts import ForwardTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor


import os
import pandas as pd

output_path = "/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/data/spanish/cml_tts_dataset_spanish_v0.1"
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Define the path to your dataset's "train.csv" file
dataset_config = BaseDatasetConfig(formatter="ljspeech", meta_file_train="train.csv", path=os.path.join(output_path))
train_csv_path = os.path.join(dataset_config.path, dataset_config.meta_file_train)

# Check if the file exists
if os.path.exists(train_csv_path):
    
    df = pd.read_csv(train_csv_path, delimiter='|')

    print(df.head())
    
    # You can perform various operations on the DataFrame to analyze the data
    # For example, you can check statistics, data types, and more.

else:
    print(f"The 'train.csv' file does not exist at path: {train_csv_path}")

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
config = FastSpeechConfig(
    run_name="fast_speech",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    precompute_num_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=30,
    use_phonemes=True,
    phonemizer="gruut",
    phoneme_language="es",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache2"),
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    min_text_len=0,
    max_text_len=500,
    min_audio_len=0,
    max_audio_len=400000,
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,
)

ap = AudioProcessor.init_from_config(config)

tokenizer, config = TTSTokenizer.init_from_config(config)

train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers

model = ForwardTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

trainer = Trainer(
      TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
  )

trainer.fit()
