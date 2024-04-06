import importlib
import sys

from flask_cors import CORS
from flask import Flask, request, jsonify
import sounddevice as sd

app = Flask(__name__)
CORS(app)

from scipy.io.wavfile import write

import base64
import subprocess
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO
import asyncio
import websockets
import json
import base64
import os
import IPython.display as ipd
import numpy as np
import torch
import re
from g2p_en import G2p
from tronduo.hparams import create_hparams
from hifigan.env import AttrDict
from hifigan.models import Generator
from tronduo.model import Tacotron2
from tronduo.layers import TacotronSTFT, STFT
from tronduo.model_util import load_model
from tronduo import text_to_sequence
from tronduo.hifigandenoiser import Denoiser
g2p = G2p()


# Load models and setup
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location='cuda')
    print("Complete.")
    return checkpoint_dict

def setup_models():
    # Your setup code for loading models and setting up parameters here...
    hparams = create_hparams()
    hparams.global_mean = None
    hparams.distributed_run = False
    hparams.prosodic = True
    hparams.speakers = True
    hparams.feat_dim = 2
    hparams.feat_max_bg = 4
    hparams.n_speakers = 3
    hparams.speaker_embedding_dim = 8

    
    path = "/home/manohargolleru2000/speech/models/tronduo/"
    iter = "70000"
    checkpoint_path = path + "checkpoint_" + iter
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cuda'))['state_dict'])
    model = model.cuda().eval()

 
    hfg_path = '/home/manohargolleru2000/speech/models/hifigan/'

    checkpoint = 2640000
    config_file = hfg_path + 'config.json'
    checkpoint_file = hfg_path + 'g_' + str(checkpoint).zfill(8)
    with open(config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)
    generator = Generator(h).to('cuda')
    state_dict_g = load_checkpoint(checkpoint_file, 'cuda')
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    denoiser = Denoiser(generator, mode='zeros')

    return model, denoiser, hparams.sampling_rate, generator


model, denoiser, sampling_rate, generator= setup_models()




def synthesize_text(transcript, speaker1=0.9, speaker2=0.1, rate=0.5, pitch=0.5):
    global model, denoiser, sampling_rate
    # Log the values of speaker1, speaker2, rate, and pitch
    print(f"Using speaker1={speaker1}, speaker2={speaker2}, rate={rate}, pitch={pitch}")
    
    # Speech synthesis
    sequence = np.array(text_to_sequence(transcript, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(torch.from_numpy(sequence)).to('cuda').long()
    speaks = torch.as_tensor([speaker1, speaker2]).unsqueeze(0).to('cuda')
    pros = torch.as_tensor([rate, pitch]).unsqueeze(0).to('cuda')  # Speech rate and pitch
    _, mel_outputs_postnet, _, _ = model.inference(sequence, speaks, pros)
    melfl = mel_outputs_postnet.float()
    y_g_hat = generator(melfl)
    audio = denoiser(y_g_hat[0], strength=0.015).squeeze()
    audio_out = audio.cpu().detach().numpy()
    audio_out = (audio_out * 32767).astype(np.int16)

    buffer = BytesIO()
    audio_segment = AudioSegment(
        audio_out.tobytes(),
        frame_rate= sampling_rate,
        sample_width=audio_out.dtype.itemsize,
        channels=1
    )

    normalized_audio = audio_segment.apply_gain(-audio_segment.max_dBFS - (-0.1))

    # Export the normalized audio data to the buffer in WAV format
    normalized_audio.export(buffer, format="wav")
    buffer.seek(0)  # Reset the buffer pointer
    return buffer

@app.route('/generate_transcript', methods=['POST'])
def generate_transcript():
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    txt = re.sub('[\!]+', ',', text)
    txt = re.sub('-', ' ', txt)
    txt = re.sub(';', '-', txt)
    txt = re.sub('\|', '. ', txt)
    phon = g2p(txt)
    for j, n in enumerate(phon):
        if n == ' ':
            phon[j] = '} {'
    transcript = '{ ' + ' '.join(phon) + ' }'
    transcript = re.sub(r' ?{ ?- ?} ?', ';', transcript)
    transcript = re.sub(r' ?{ ?, ?} ?', ',', transcript)
    transcript = re.sub(r' ?{ ?\. ?} ?', '.', transcript)
    transcript = re.sub(r' ?{ ?\? ?} ?', '?', transcript)
    transcript = re.sub(r'{ ?', '{', transcript)
    transcript = re.sub(r' ?}', '}', transcript)
    if transcript.strip()[-1:] == '}':
        transcript = transcript.strip() + '.'

    return jsonify({'transcript': transcript})

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.get_json()
    transcript = data.get('transcript')
    speaker1 = data.get('Speaker1', 0.9)  # Use default value if not provided
    speaker2 = data.get('Speaker2', 0.1)  # Use default value if not provided
    pitch = data.get('pitch', 0.5)  # Use default value if not provided
    rate = data.get('rate', 0.5)  # Use default value if not provided
    volume = data.get('volume', 1)  # Use default value if not provided

    if not transcript:
        return jsonify({'error': 'transcript not provided'}), 400

    # Log the received values
    print(f'Received request to synthesize: "{transcript}"')
    print(f'Speaker1: {speaker1}, Speaker2: {speaker2}, pitch: {pitch}, rate: {rate}, volume: {volume}')

    # Get audio data from synthesize_text function
    audio_buffer = synthesize_text(transcript, speaker1, speaker2, pitch, rate)

    # Log that audio data is being sent
    print('Sending audio data to client...')

    # Set the Content-Type header to indicate WAV audio
    response = app.response_class(
        response=audio_buffer.getvalue(),
        status=200,
        mimetype='audio/wav'
    )

    return response



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
