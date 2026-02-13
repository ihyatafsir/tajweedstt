#!/usr/bin/env python3
"""
CTC Forced Aligner + Physics for Surah 91 (Al-Balad)
Uses ctc-forced-aligner (wav2vec CTC) + TajweedSST physics refinement.

Pipeline:
1. CTC Alignment: wav2vec forced alignment for letter timing
2. Tajweed Parser: Map letters to Tajweed rules
3. Physics Validation: Validate with acoustic physics
4. Export: MahQuranApp format

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python3 ctc_align_91.py
"""
import json
import torch
import sys
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

sys.path.insert(0, str(Path(__file__).parent))
from src.tajweed_parser import TajweedParser, TajweedType, PhysicsCheck
from src.physics_validator import PhysicsValidator, ValidationStatus
from src.duration_model import DurationModel, MaddType

import librosa

# Config
SURAH = 90
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
OUTPUT_DIR = PROJECT_ROOT / "public/data/abdul_basit"
AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_090.mp3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4


def load_quran_text(surah_num: int) -> str:
    """Load Quran text from verses_v4.json"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        all_verses = json.load(f)
    verses = all_verses.get(str(surah_num), [])
    return ' '.join(v.get('text', '') for v in verses)


def run_ctc_alignment(text: str):
    """Run CTC forced alignment"""
    print("\n[1] Loading wav2vec alignment model...")
    alignment_model, alignment_tokenizer = load_alignment_model(
        DEVICE,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    )
    
    print("\n[2] Loading audio...")
    audio_waveform = load_audio(str(AUDIO_PATH), alignment_model.dtype, alignment_model.device)
    
    print("\n[3] Generating CTC emissions...")
    emissions, stride = generate_emissions(
        alignment_model, audio_waveform, batch_size=BATCH_SIZE
    )
    print(f"    Emissions shape: {emissions.shape}")
    
    print("\n[4] Preprocessing text...")
    tokens_starred, text_starred = preprocess_text(
        text,
        romanize=True,
        language="ara",
    )
    
    print("\n[5] Getting alignments...")
    segments, scores, blank_token = get_alignments(
        emissions, tokens_starred, alignment_tokenizer,
    )
    
    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    
    print(f"    Got {len(word_timestamps)} word alignments")
    
    # Cleanup GPU
    del alignment_model
    torch.cuda.empty_cache()
    
    return word_timestamps


def convert_to_char_timings(word_timestamps):
    """Convert word timestamps to character-level timing"""
    char_timings = []
    word_idx = 0
    
    for wt in word_timestamps:
        word = wt['text']
        start = wt['start']
        end = wt['end']
        duration = end - start
        char_dur = duration / len(word) if word else 0
        
        word_has_chars = False
        for i, char in enumerate(word):
            if not char.isspace():
                word_has_chars = True
                char_timings.append({
                    "char": char,
                    "start": round(start + i * char_dur, 3),
                    "end": round(start + (i + 1) * char_dur, 3),
                    "idx": len(char_timings),
                    "wordIdx": word_idx
                })
        
        if word_has_chars:
            word_idx += 1
    
    return char_timings


def apply_physics(char_timings, text):
    """Apply Tajweed parsing and physics validation"""
    print("\n[6] Parsing Tajweed rules...")
    parser = TajweedParser()
    
    # Get all letter tags
    all_tags = []
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        verses = json.load(f).get(str(SURAH), [])
    
    for verse in verses:
        word_tags = parser.parse_text(verse['text'])
        for word_tag in word_tags:
            for letter in word_tag.letters:
                all_tags.append({
                    'char': letter.char_visual,
                    'tajweed_type': letter.tajweed_type,
                    'physics_check': letter.physics_check,
                    'madd_count': letter.madd_count
                })
    
    print(f"    Tajweed tags: {len(all_tags)}")
    
    # Load audio for physics
    print("\n[7] Loading audio for physics...")
    audio, sr = librosa.load(str(AUDIO_PATH), sr=22050)
    physics = PhysicsValidator(sample_rate=sr)
    duration_model = DurationModel()
    
    # Calibrate
    vowels = [t['end'] - t['start'] for t in char_timings if 0.05 <= (t['end'] - t['start']) <= 0.15]
    if vowels:
        duration_model.calibrate_from_samples("Abdul_Basit", vowels)
        print(f"    Harakat: {duration_model.calibration.harakat_base_ms:.1f}ms")
    
    # Apply physics
    print("\n[8] Applying physics validation...")
    stats = {'total': 0, 'validated': 0, 'passed': 0, 'marginal': 0, 'failed': 0}
    
    for i, entry in enumerate(char_timings):
        stats['total'] += 1
        
        if i < len(all_tags):
            tag = all_tags[i]
            entry['tajweed'] = tag['tajweed_type'].value
            
            if tag['physics_check'] != PhysicsCheck.NONE:
                stats['validated'] += 1
                start, end = entry['start'], entry['end']
                
                try:
                    check = tag['physics_check']
                    
                    if check == PhysicsCheck.CHECK_RMS_BOUNCE:
                        val = physics.validate_qalqalah(audio, start, end)
                    elif check == PhysicsCheck.CHECK_DURATION:
                        val = physics.validate_madd(audio, start, end, tag['madd_count'] or 2)
                    elif check == PhysicsCheck.CHECK_GHUNNAH:
                        val = physics.validate_ghunnah(audio, start, end)
                    elif check == PhysicsCheck.CHECK_FORMANT_F2:
                        val = physics.validate_tafkheem(audio, start, end)
                    else:
                        val = None
                    
                    if val:
                        entry['physics'] = val.status.value
                        entry['score'] = float(round(val.score, 2))
                        
                        if val.status == ValidationStatus.PASS:
                            stats['passed'] += 1
                        elif val.status == ValidationStatus.MARGINAL:
                            stats['marginal'] += 1
                        else:
                            stats['failed'] += 1
                except Exception:
                    pass
    
    return char_timings, stats


def main():
    print("=" * 60)
    print(f"CTC + Physics Pipeline: Surah {SURAH} (Al-Balad)")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Get text
    text = load_quran_text(SURAH)
    print(f"\nText length: {len(text)} chars")
    
    # Run CTC alignment
    word_timestamps = run_ctc_alignment(text)
    
    # Convert to char timings
    char_timings = convert_to_char_timings(word_timestamps)
    print(f"\n    Total chars: {len(char_timings)}")
    
    # Apply physics
    char_timings, stats = apply_physics(char_timings, text)
    
    # Print stats
    print(f"\n[9] Statistics:")
    print(f"    Total: {stats['total']}")
    print(f"    Validated: {stats['validated']}")
    print(f"    ✓ Passed: {stats['passed']}")
    print(f"    ~ Marginal: {stats['marginal']}")
    print(f"    ✗ Failed: {stats['failed']}")
    
    if stats['validated'] > 0:
        rate = (stats['passed'] + stats['marginal']) / stats['validated'] * 100
        print(f"    Pass Rate: {rate:.1f}%")
    
    # Save
    output_path = OUTPUT_DIR / f"letter_timing_{SURAH}_ctc.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(char_timings, f, ensure_ascii=False, indent=2)
    print(f"\n[10] Saved: {output_path}")
    
    # Show sample
    print("\n=== First 15 characters ===")
    for ct in char_timings[:15]:
        tj = ct.get('tajweed', 'None')
        ph = ct.get('physics', '-')
        print(f"  {ct['idx']:3d}: '{ct['char']}' @ {ct['start']:.3f}s | {tj} | {ph}")
    
    print("\n" + "=" * 60)
    print("✓ CTC + Physics Pipeline complete!")
    print(f"  Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
