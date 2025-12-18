
from TTS.api import TTS
import time
import json
import random
import time
from pathlib import Path
import streamlit as st
import time
import html

import gradio as gr
import uuid
import torch
import librosa
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from scipy.io.wavfile import write

params = {
    "activate": True,
    "autoplay": True,
    "show_text": False,
    "remove_trailing_dots": False,
    "voice": "Rogger.wav",
    "language": "English",
    "model_name": "tts_models/multilingual/multi-dataset/xtts_v2",
    # "model_path":"./models/",
    # "config_path":"./models/config.json"
}

SUPPORTED_FORMATS = ['wav']
SAMPLE_RATE = 16000

speakers = {p.stem: str(p) for p in list(Path('targets').iterdir())}

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# device = torch.device('cuda:0')

# device = torch.device('cuda:0')
import os
os.makedirs(os.path.join(".", "targets"), exist_ok=True)
os.makedirs(os.path.join(".", "outputs"), exist_ok=True)

@st.cache_resource
# @st.experimental_singleton
def load_model():
    global tts
    print("[XTTS] Loading XTTS...")
    tts = TTS(model_name=params["model_name"]).to(device)
    # model_path=params["model_path"],
    # config_path=params["config_path"]).
    return tts

tts=load_model()

def get_available_voices():
    return sorted([voice.name for voice in Path(f"{this_dir}/targets").glob("*.wav")])

def random_sentence():
    with open(Path("harvard_sentences.txt")) as f:
        return random.choice(list(f))

st.title("TTS based Voice Cloning in 16 Languages.")
# st.image('logo.png', width=150)

st.header('Text to speech generation')

this_dir = str(Path(__file__).parent.resolve())
languages=None
with open(Path(f"{this_dir}/languages.json"), encoding='utf8') as f:
    languages = json.load(f)

with st.sidebar:
    voice_list=get_available_voices()
    print (voice_list)
    st.title("Text to Voice")
    english = st.radio(
        label="Choose your language", options=languages, index=0, horizontal=True)

    default_speaker_name = "Rogger"
    speaker_name = st.selectbox('Select target speaker:', options=[None] + list(speakers.keys()),
    index=[key for key in speakers.keys()].index(default_speaker_name) + 1 if default_speaker_name in speakers else 0)

    wav_tgt=None
    if speaker_name is not None:
        wav_tgt, _ = librosa.load(speakers[speaker_name],sr=22000)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        st.write('Selected Target:')
        st.audio(wav_tgt, sample_rate=22000)

    # Upload the WAV file

    text = st.text_area('Enter text to convert to audio format',
    value="Hello")
    speed = st.slider('Speed', 0.1, 1.99, 0.8, 0.01)

def gen_voice(string,spk):
    string = html.unescape(string)
    # Generate a short UUID
    short_uuid = str(uuid.uuid4())[:8]
    fl_name='outputs/' + spk + "-" + short_uuid +'.wav'
    output_file = Path(fl_name)

    # Get the speaker WAV file path
    speaker_wav_path = f"{this_dir}/targets/" + spk + ".wav"
    language_code = languages[english]

    # Generate TTS
    tts.tts_to_file(
        text=string,
        speed=speed,
        file_path=output_file,
        speaker_wav=[speaker_wav_path],
        language=language_code
    )

    # Save latent embeddings for Mantella integration
    save_latents_for_mantella(spk, speaker_wav_path, language_code)

    return output_file


def save_latents_for_mantella(speaker_name, speaker_wav_path, language_code):
    """Save latent embeddings to Mantella-compatible format"""
    try:
        import json

        # Map language names to codes
        language_map = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Portuguese': 'pt',
            'Polish': 'pl',
            'Turkish': 'tr',
            'Russian': 'ru',
            'Dutch': 'nl',
            'Czech': 'cs',
            'Arabic': 'ar',
            'Chinese (Simplified)': 'zh-cn',
            'Japanese': 'ja',
            'Hungarian': 'hu',
            'Korean': 'ko',
            'Hindi': 'hi'
        }

        # Get language code
        lang_code = language_map.get(language_code, 'en')

        # Create latent speaker directory structure
        latent_base_dir = Path("latent_speaker_folder")
        latent_lang_dir = latent_base_dir / lang_code
        latent_lang_dir.mkdir(parents=True, exist_ok=True)

        # Debug: Print TTS object structure
        print(f"TTS object type: {type(tts)}")
        print(f"TTS object attributes: {dir(tts)}")

        # Try multiple ways to access the conditioning latents method
        gpt_cond_latent = None
        speaker_embedding = None

        # Method 1: Direct access to synthesizer model
        if hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'tts_model'):
            model = tts.synthesizer.tts_model
            print(f"Found synthesizer.tts_model: {type(model)}")
            if hasattr(model, 'get_conditioning_latents'):
                print("Using synthesizer.tts_model.get_conditioning_latents")
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=speaker_wav_path,
                    gpt_cond_len=30,
                    max_ref_length=60
                )

        # Method 2: Access model through synthesizer
        elif hasattr(tts, 'synthesizer') and hasattr(tts.synthesizer, 'model'):
            model = tts.synthesizer.model
            print(f"Found synthesizer.model: {type(model)}")
            if hasattr(model, 'get_conditioning_latents'):
                print("Using synthesizer.model.get_conditioning_latents")
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=speaker_wav_path,
                    gpt_cond_len=30,
                    max_ref_length=60
                )

        # Method 3: Check if tts has model attribute directly
        elif hasattr(tts, 'model'):
            model = tts.model
            print(f"Found direct model: {type(model)}")
            if hasattr(model, 'get_conditioning_latents'):
                print("Using direct model.get_conditioning_latents")
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
                    audio_path=speaker_wav_path,
                    gpt_cond_len=30,
                    max_ref_length=60
                )

        if gpt_cond_latent is None or speaker_embedding is None:
            raise AttributeError(f"Cannot find get_conditioning_latents method. Available TTS attributes: {list(dir(tts))}")

        # Prepare data in Mantella format (preserve original tensor dimensions)
        latent_data = {
            "gpt_cond_latent": gpt_cond_latent.cpu().tolist(),
            "speaker_embedding": speaker_embedding.cpu().tolist()
        }

        # Save to JSON file
        latent_file_path = latent_lang_dir / f"{speaker_name.lower()}.json"
        with open(latent_file_path, 'w') as f:
            json.dump(latent_data, f)

        st.success(f"✅ Saved latent embeddings to: {latent_file_path}")
        print(f"Latent embeddings saved for {speaker_name} in {lang_code}: {latent_file_path}")

    except Exception as e:
        st.warning(f"⚠️ Could not save latent embeddings: {str(e)}")
        print(f"Error saving latents: {e}")
        # Print full traceback for debugging
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")

st.caption ("Optional Microphone Recording. Download and rename your recording before using.")
audio_bytes = audio_recorder()
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

# Upload the WAV file
st.caption ("For the audio file, use the name of your Target, for instance ABIDA.wav")
new_tgt = st.file_uploader('Upload a new TARGET audio WAV file:', type=SUPPORTED_FORMATS, accept_multiple_files=False)
if new_tgt is not None:
    # Get the original file name
    file_name = new_tgt.name

    # Save the file to the file system
    file_path = os.path.join("./targets/", file_name)
    st.info(f"Original file name: {file_name}")

    # Extract the file name without the extension
    file_name_without_extension = os.path.splitext(file_name)[0]

    # Use librosa to load and process the WAV file
    wav_tgt, _ = librosa.load(new_tgt, sr=22000)
    wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

    # Use scipy.io.wavfile.write to save the processed WAV file
    write('./targets/' + file_name_without_extension + '.wav', 22000, wav_tgt)

    st.success(f"New target saved successfully to {file_path}")

if st.button('Convert'):
    # Run TTS
    st.success('Converting ... please wait ...')
    output_file=gen_voice(text, speaker_name)
    # tts.tts_to_file(text=text, speed=speed, speaker=speaker, file_path="out.wav")

    st.write(f'Target voice:'+speaker_name)
    # st.success('Converted to audio successfully')

    audio_file = open(output_file, 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')
    # st.success("You can now play the audio by clicking on the play button.")
