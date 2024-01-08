def ljspeech_test(root_path, meta_file, **kwargs):  # pylint: disable=unused-argum>    """Normalizes the LJSpeech meta data file for TTS testing
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        speaker_id = 0
        for idx, line in enumerate(ttf):
            # 2 samples per speaker to avoid eval split issues
            if idx % 2 == 0:
                speaker_id += 1
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[3]
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": f"ljspeech->            )
    return items

