#!/usr/bin/env python3
"""
CTC Forced Aligner for Surah 90 (Al-Balad)
Uses ctc-forced-aligner v0.3.0 from GitHub for word-level alignment.
Based on MahQuranApp/scripts/ctc_quran_aligner.py
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

# Config
SURAH = 90
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
OUTPUT_DIR = PROJECT_ROOT / "public/data"
AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_090.mp3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

def load_quran_text(surah_num: int) -> str:
    """Load Quran text from verses_v4.json"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    verses = all_verses.get(str(surah_num), [])
    return ' '.join(v.get('text', '') for v in verses)

def main():
    print("=" * 60)
    print(f"CTC Forced Aligner for Surah {SURAH} (Al-Balad)")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # 1. Load alignment model
    print("\n[1] Loading alignment model...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    print("    Model loaded.")
    
    # 2. Load audio
    print("\n[2] Loading audio...")
    audio_waveform = load_audio(str(AUDIO_PATH), alignment_model.dtype, alignment_model.device)
    print(f"    Audio loaded.")
    
    # 3. Get Quran text
    text = load_quran_text(SURAH)
    print(f"\n[3] Text length: {len(text)} chars")
    print(f"    First 60: {text[:60]}...")
    
    # 4. Generate emissions
    print("\n[4] Generating emissions...")
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=BATCH_SIZE
    )
    print(f"    Emissions shape: {emissions.shape}")
    
    # 5. Preprocess text
    print("\n[5] Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language="ara",
    )
    
    # 6. Get alignments
    print("\n[6] Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer,
    )
    
    # 7. Get spans
    spans = get_spans(tokens_starred, segments, blank_token)
    
    # 8. Post-process results
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    print(f"    Got {len(word_timestamps)} word alignments")
    
    # 9. Convert to character-level timing (seconds format)
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
                    "char": char,
                    "start": round(start + i * char_dur, 3),
                    "end": round(start + (i + 1) * char_dur, 3),
                    "idx": len(char_timings)
                })
    
    print(f"\n[7] Total chars: {len(char_timings)}")
    
    # 10. Save output
    output_path = OUTPUT_DIR / f"letter_timing_{SURAH}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(char_timings, f, ensure_ascii=False, indent=2)
    
    print(f"\n[8] Saved to {output_path}")
    
    # Print first 20 for verification
    print("\n=== First 20 characters ===")
    for ct in char_timings[:20]:
        dur_ms = (ct['end'] - ct['start']) * 1000
        print(f"  {ct['idx']:3d}: '{ct['char']}' @ {ct['start']:.3f}s - {ct['end']:.3f}s ({dur_ms:.0f}ms)")
    
    print("\n" + "=" * 60)
    print("✓ CTC Alignment complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
