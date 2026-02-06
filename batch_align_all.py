#!/usr/bin/env python3
"""
Batch CTC Alignment for All Abdul Basit Surahs
Processes all 114 surahs with the full pipeline:
1. CTC forced alignment (wav2vec2)
2. Grapheme matching (App.tsx compatible)
3. Export to MahQuranApp format

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python batch_align_all.py
"""
import json
import sys
import time
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

# Config
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
AUDIO_DIR = PROJECT_ROOT / "public/audio/abdul_basit"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

# Exact same DIACRITICS as App.tsx line 176
DIACRITICS = set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ', 'ۖ', 'ۗ', 'ۘ', 'ۙ', 'ۚ', 'ۛ', 'ۜ', 'ٔ', 'ٓ', 'ـ'])


def is_diacritic(ch):
    """Match App.tsx splitIntoGraphemes exactly"""
    return ch in DIACRITICS or (0x064B <= ord(ch) <= 0x0652) or (0x0610 <= ord(ch) <= 0x061A)


def split_into_graphemes(text):
    """Exact same logic as App.tsx splitIntoGraphemes"""
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


def load_quran_text(all_verses, surah_num):
    """Load Quran text for a surah"""
    verses = all_verses.get(str(surah_num), [])
    return ' '.join(v.get('text', '') for v in verses)


def get_grapheme_list(all_verses, surah_num):
    """Get graphemes with ayah info matching App.tsx rendering"""
    verses = all_verses.get(str(surah_num), [])
    grapheme_list = []
    for v in verses:
        for word in v['text'].split():
            for g in split_into_graphemes(word):
                grapheme_list.append({'char': g, 'ayah': v['ayah']})
    return grapheme_list


def process_surah(surah_num, alignment_model, alignment_tokenizer, all_verses):
    """Process a single surah through the full pipeline"""
    audio_path = AUDIO_DIR / f"surah_{surah_num:03d}.mp3"
    output_path = OUTPUT_DIR / f"letter_timing_{surah_num}.json"

    if not audio_path.exists():
        return None, "No audio file"

    text = load_quran_text(all_verses, surah_num)
    if not text.strip():
        return None, "No verse text"

    grapheme_list = get_grapheme_list(all_verses, surah_num)

    try:
        # Step 1: Load audio
        audio_waveform = load_audio(str(audio_path), alignment_model.dtype, alignment_model.device)

        # Step 2: Generate CTC emissions
        emissions, stride = generate_emissions(
            alignment_model, audio_waveform, batch_size=BATCH_SIZE
        )

        # Step 3: Preprocess text  
        tokens_starred, text_starred = preprocess_text(
            text, romanize=True, language="ara",
        )

        # Step 4: Get alignments
        segments, scores, blank_token = get_alignments(
            emissions, tokens_starred, alignment_tokenizer,
        )

        # Step 5: Get spans & post-process
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Step 6: Expand to character-level
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

        # Step 7: Map CTC chars to graphemes
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
                'duration': e - s,
                'wordIdx': gi // 4,
                'weight': 1.0
            })

        # Save
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timing, f, ensure_ascii=False, indent=2)

        return len(timing), f"OK ({len(grapheme_list)} graphemes)"

    except Exception as ex:
        return None, f"Error: {ex}"


def main():
    start_time = time.time()
    print("=" * 60)
    print("Batch CTC Alignment - Abdul Basit (All 114 Surahs)")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Load model once
    print("\n[1] Loading wav2vec alignment model...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    print("    Model loaded.")

    # Load all verses
    print("[2] Loading verses...")
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    print(f"    Loaded {len(all_verses)} surahs")

    # Process each surah
    results = []
    for surah_num in range(1, 115):
        elapsed = time.time() - start_time
        print(f"\n[Surah {surah_num:03d}/114] ({elapsed:.0f}s elapsed)...")

        count, status = process_surah(
            surah_num, alignment_model, alignment_tokenizer, all_verses
        )
        results.append((surah_num, count, status))

        if count:
            print(f"    ✓ {count} letters - {status}")
        else:
            print(f"    ✗ {status}")

    # Summary
    elapsed = time.time() - start_time
    ok = sum(1 for _, c, _ in results if c)
    fail = sum(1 for _, c, _ in results if not c)

    print("\n" + "=" * 60)
    print(f"BATCH COMPLETE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  ✓ Success: {ok}/114")
    print(f"  ✗ Failed:  {fail}/114")
    print("=" * 60)

    # Cleanup
    del alignment_model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
