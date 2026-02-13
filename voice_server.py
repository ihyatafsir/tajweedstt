#!/usr/bin/env python3
"""
Real-Time Voice CTC Alignment Server
Receives audio chunks from browser, runs CTC alignment,
returns letter-level timestamps for dual-layer highlighting.

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python voice_server.py
"""
import json
import io
import time
import torch
import numpy as np
import librosa
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

app = Flask(__name__)
CORS(app)

# Config
DEVICE = "cpu"
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"

# Exact same DIACRITICS as App.tsx
DIACRITICS = set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', 'ۖ', 'ۗ', 'ۘ', 'ۙ', 'ۚ', 'ۛ', 'ۜ', 'ٔ', 'ٓ', 'ـ'])

# Global model (loaded once)
alignment_model = None
alignment_tokenizer = None
all_verses = None


def is_diacritic(ch):
    return ch in DIACRITICS or (0x064B <= ord(ch) <= 0x0652) or (0x0610 <= ord(ch) <= 0x061A)


def split_into_graphemes(text):
    graphemes = []
    current = ''
    for ch in text:
        if ch == ' ':
            if current:
                graphemes.append(current)
                current = ''
        elif is_diacritic(ch) and current:
            current += ch
        else:
            if current:
                graphemes.append(current)
            current = ch
    if current:
        graphemes.append(current)
    return graphemes


def load_models():
    """Load CTC model and verse data on startup"""
    global alignment_model, alignment_tokenizer, all_verses

    print("[1] Loading wav2vec alignment model...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE, dtype=torch.float32
    )
    print("    Model loaded.")

    print("[2] Loading verses...")
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    print(f"    Loaded {len(all_verses)} surahs")


def align_audio(audio_data, sample_rate, surah_num, start_ayah=1, end_ayah=None):
    """
    Align audio data to Quran text and return letter timestamps.
    
    Args:
        audio_data: numpy array of audio samples
        sample_rate: audio sample rate
        surah_num: surah number (1-114)
        start_ayah: starting ayah
        end_ayah: ending ayah (None = all)
    
    Returns:
        list of {idx, char, ayah, start_ms, end_ms} matching graphemes
    """
    verses = all_verses.get(str(surah_num), [])
    if not verses:
        return []

    # Filter ayahs
    if end_ayah:
        verses = [v for v in verses if start_ayah <= v['ayah'] <= end_ayah]
    else:
        verses = [v for v in verses if v['ayah'] >= start_ayah]

    text = ' '.join(v['text'] for v in verses)
    if not text.strip():
        return []

    # Build grapheme list
    grapheme_list = []
    for v in verses:
        for word in v['text'].split():
            for g in split_into_graphemes(word):
                grapheme_list.append({'char': g, 'ayah': v['ayah']})

    # Resample to 16kHz for wav2vec
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

    # Convert to tensor
    audio_tensor = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    try:
        # CTC alignment
        emissions, stride = generate_emissions(
            alignment_model, audio_tensor, batch_size=1
        )

        tokens_starred, text_starred = preprocess_text(
            text, romanize=True, language="ara"
        )

        segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Expand to character-level
        char_timings = []
        for wt in word_timestamps:
            word = wt['text']
            start = wt['start']
            end = wt['end']
            dur = end - start
            char_dur = dur / len(word) if word else 0
            for i, char in enumerate(word):
                if not char.isspace():
                    char_timings.append({
                        'start': start + i * char_dur,
                        'end': start + (i + 1) * char_dur,
                    })

        # Map to graphemes
        timing = []
        ci = 0
        for gi, ginfo in enumerate(grapheme_list):
            g = ginfo['char']
            s, e = None, None
            for _ in range(len(g)):
                if ci < len(char_timings):
                    if s is None:
                        s = int(char_timings[ci]['start'] * 1000)
                    e = int(char_timings[ci]['end'] * 1000)
                    ci += 1
            if s is None:
                s = timing[-1]['end'] if timing else 0
                e = s + 100

            timing.append({
                'idx': gi,
                'char': g,
                'ayah': ginfo['ayah'],
                'start': s,
                'end': e,
            })

        return timing

    except Exception as ex:
        print(f"Alignment error: {ex}")
        return []


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': alignment_model is not None,
        'surahs_loaded': len(all_verses) if all_verses else 0,
    })


@app.route('/align', methods=['POST'])
def align():
    """
    Receive audio + surah info, return letter timestamps.
    
    POST body (multipart):
        audio: WAV/PCM audio file
        surah: surah number
        start_ayah: optional starting ayah
        end_ayah: optional ending ayah
    
    OR POST body (JSON):
        audio_b64: base64 encoded audio
        sample_rate: sample rate
        surah: surah number
    """
    start_time = time.time()

    if 'audio' in request.files:
        # File upload
        audio_file = request.files['audio']
        surah_num = int(request.form.get('surah', 1))
        start_ayah = int(request.form.get('start_ayah', 1))
        end_ayah = request.form.get('end_ayah')
        end_ayah = int(end_ayah) if end_ayah else None

        # Read audio
        audio_bytes = audio_file.read()
        audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

    elif request.is_json:
        import base64
        data = request.json
        surah_num = data.get('surah', 1)
        start_ayah = data.get('start_ayah', 1)
        end_ayah = data.get('end_ayah')

        audio_b64 = data.get('audio_b64', '')
        sr = data.get('sample_rate', 16000)
        audio_bytes = base64.b64decode(audio_b64)
        audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

    else:
        return jsonify({'error': 'No audio provided'}), 400

    # Run alignment
    timing = align_audio(audio_data, sr, surah_num, start_ayah, end_ayah)

    elapsed = time.time() - start_time
    return jsonify({
        'timing': timing,
        'count': len(timing),
        'elapsed_ms': int(elapsed * 1000),
        'surah': surah_num,
    })


@app.route('/align_realtime', methods=['POST'])
def align_realtime():
    """
    Lightweight real-time alignment endpoint for mobile.
    
    POST body (JSON):
        audio_b64: base64 encoded PCM float32 audio (16kHz mono)
        sample_rate: sample rate (default 16000)
        surah: surah number
        start_ayah: optional starting ayah
    
    Returns:
        {letter_idx: int, total_letters: int, elapsed_ms: int}
    """
    import base64
    start_time = time.time()

    if not request.is_json:
        return jsonify({'error': 'JSON required'}), 400

    data = request.json
    surah_num = data.get('surah', 1)
    start_ayah = data.get('start_ayah', 1)
    audio_b64 = data.get('audio_b64', '')
    sr = data.get('sample_rate', 16000)

    if not audio_b64:
        return jsonify({'error': 'No audio provided'}), 400

    audio_bytes = base64.b64decode(audio_b64)
    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)

    if len(audio_data) < 1600:  # < 100ms of audio
        return jsonify({'letter_idx': -1, 'total_letters': 0, 'elapsed_ms': 0})

    # Run full alignment
    timing = align_audio(audio_data, sr, surah_num, start_ayah)

    if not timing:
        return jsonify({'letter_idx': -1, 'total_letters': 0, 'elapsed_ms': 0})

    # Find current letter index based on audio duration
    audio_duration_ms = int(len(audio_data) / sr * 1000)
    letter_idx = 0
    for t in timing:
        if t['end'] <= audio_duration_ms:
            letter_idx = t['idx']
        else:
            break

    elapsed = time.time() - start_time
    return jsonify({
        'letter_idx': letter_idx,
        'total_letters': len(timing),
        'elapsed_ms': int(elapsed * 1000),
    })


if __name__ == '__main__':
    load_models()
    print("\n" + "=" * 50)
    print("Voice CTC Server ready on http://0.0.0.0:5174")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5174, debug=False)
