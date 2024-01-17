import argparse
import torch
import sys
# Path where the TTS modules are located
tts_module_path = '/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/adrelingyte/TTS'
# Append this path to sys.path
sys.path.append(tts_module_path)
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.config import load_config
import soundfile as sf
from tqdm import tqdm
from glob import glob
import os

'''

'''



USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_vocoder(checkpoint_filepath: str, config_filepath: str) -> None:
    vocoder_config = load_config(config_filepath)
    vocoder_model = setup_vocoder_model(vocoder_config)
    vocoder_model.load_checkpoint(vocoder_config, checkpoint_filepath, eval=True)
    if USE_CUDA:
        vocoder_model.cuda()
    return vocoder_model
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_vocoder', default='WaveGrad/run-December-18-2023_02+24PM-f579dfe/best_model.pth', help='Checkpoint pth filepath')
    parser.add_argument('--config_vocoder', default='WaveGrad/run-December-18-2023_02+24PM-f579dfe/config.json', help='Config json filepath')
    parser.add_argument('--input_dir', default='cml_tts_dataset_portuguese_v0.1/dev/audio/9056/7087/', help='Input folder')
    parser.add_argument('--output_dir', default='output_vocoder', help='Output folder')
    args = parser.parse_args()

    vocoder_model = load_vocoder(args.checkpoint_vocoder, args.config_vocoder)
    config = load_config(args.config_vocoder)
    ap = AudioProcessor(**config.audio.to_dict())

    os.makedirs(args.output_dir, exist_ok=True)
    for audio_filepath in tqdm(glob(os.path.join(args.input_dir, "*.wav"))):
        filename = os.path.basename(audio_filepath)
        waveform = ap.load_wav(audio_filepath)
        mel_spec = ap.melspectrogram(waveform)
        mel_spec = torch.from_numpy(mel_spec)
        pred_waveform = vocoder_model.inference(mel_spec.unsqueeze(0).cuda())
        filepath = os.path.join(args.output_dir, filename.replace(".flac", ".wav"))
        ap.save_wav(pred_waveform.cpu().squeeze().numpy(), filepath, ap.sample_rate)


if __name__ == "__main__":
    main()
