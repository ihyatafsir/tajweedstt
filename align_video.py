#!/usr/bin/env python3
"""
CTC Forced Aligner for Video Audio
Aligns extracted audio from a video against known Quran text
to produce letter-level timing JSON for overlay rendering.

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python align_video.py
"""
import json
import torch
from pathlib import Path
from ctc_forced_aligner import (
    load_audio,
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)

# === CONFIG ===
AUDIO_PATH = Path("/home/absolut7/Downloads/twitter_audio.wav")
OUTPUT_PATH = Path("/home/absolut7/Downloads/video_letter_timing.json")
VERSES_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json")
SURAH_NUM = 9  # At-Tawbah
AYAH_START = 50
AYAH_END = 55
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# Exact DIACRITICS set (same as App.tsx)
DIACRITICS = set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', 'ۖ', 'ۗ', 'ۘ', 'ۙ', 'ۚ', 'ۛ', 'ۜ', 'ٔ', 'ٓ', 'ـ'])


def is_diacritic(ch):
    return ch in DIACRITICS or (0x064B <= ord(ch) <= 0x0652) or (0x0610 <= ord(ch) <= 0x061A)


def split_into_graphemes(text):
    """Split Arabic text into graphemes (base char + diacritics)"""
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


def main():
    print("=" * 60)
    print(f"CTC Forced Aligner — Video Audio")
    print(f"Surah {SURAH_NUM}, Ayat {AYAH_START}-{AYAH_END}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 1. Load verses
    print("\n[1] Loading Quran text...")
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)

    verses = all_verses.get(str(SURAH_NUM), [])
    selected = [v for v in verses if AYAH_START <= v['ayah'] <= AYAH_END]
    full_text = ' '.join(v['text'] for v in selected)
    print(f"    {len(selected)} ayat, {len(full_text)} chars")
    print(f"    First 80: {full_text[:80]}...")

    # Build grapheme list with ayah info
    grapheme_list = []
    word_idx = 0
    for v in selected:
        words = v['text'].split()
        for word in words:
            for g in split_into_graphemes(word):
                grapheme_list.append({
                    'char': g,
                    'ayah': v['ayah'],
                    'wordIdx': word_idx
                })
            word_idx += 1

    print(f"    {len(grapheme_list)} graphemes total")

    # 2. Load alignment model
    print("\n[2] Loading wav2vec alignment model...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    print("    Model loaded.")

    # 3. Load audio
    print("\n[3] Loading audio...")
    audio_waveform = load_audio(str(AUDIO_PATH), alignment_model.dtype, alignment_model.device)
    print(f"    Audio loaded.")

    # 4. Generate CTC emissions
    print("\n[4] Generating emissions...")
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=BATCH_SIZE
    )
    print(f"    Emissions shape: {emissions.shape}")

    # 5. Preprocess text
    print("\n[5] Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        full_text, romanize=True, language="ara",
    )

    # 6. Get alignments
    print("\n[6] Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer,
    )

    # 7. Get spans & post-process
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    print(f"    Got {len(word_timestamps)} word alignments")

    # 8. Expand to character-level
    char_timings = []
    for wt in word_timestamps:
        word = wt['text']
        start = wt['start']
        end = wt['end']
        duration = end - start
        char_dur = duration / len(word) if word else 0
        for i, char in enumerate(word):
            if not char.isspace():
                char_timings.append({
                    'start': start + i * char_dur,
                    'end': start + (i + 1) * char_dur,
                })

    # 9. Map CTC chars to graphemes
    timing = []
    ci = 0
    for gi, ginfo in enumerate(grapheme_list):
        g = ginfo['char']
        s, e = None, None
        for _ in range(len(g)):
            if ci < len(char_timings):
                if s is None:
                    s = char_timings[ci]['start']
                e = char_timings[ci]['end']
                ci += 1
        if s is None:
            s = timing[-1]['end_s'] if timing else 0.0
            e = s + 0.1

        timing.append({
            'idx': gi,
            'char': g,
            'ayah': ginfo['ayah'],
            'wordIdx': ginfo['wordIdx'],
            'start_s': round(s, 4),
            'end_s': round(e, 4),
            'duration_ms': round((e - s) * 1000, 1),
        })

    print(f"\n[7] Total graphemes timed: {len(timing)}")

    # 10. Save output
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(timing, f, ensure_ascii=False, indent=2)

    print(f"\n[8] Saved to {OUTPUT_PATH}")

    # Print first 20 for verification
    print("\n=== First 20 graphemes ===")
    for t in timing[:20]:
        print(f"  {t['idx']:3d}: '{t['char']}' ayah={t['ayah']} @ {t['start_s']:.3f}s-{t['end_s']:.3f}s ({t['duration_ms']:.0f}ms)")

    # Print last 10
    print("\n=== Last 10 graphemes ===")
    for t in timing[-10:]:
        print(f"  {t['idx']:3d}: '{t['char']}' ayah={t['ayah']} @ {t['start_s']:.3f}s-{t['end_s']:.3f}s ({t['duration_ms']:.0f}ms)")

    print("\n" + "=" * 60)
    print("✓ CTC Alignment complete!")
    print("=" * 60)

    # Cleanup
    del alignment_model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
