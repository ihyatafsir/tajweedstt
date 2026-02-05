#!/usr/bin/env python3
"""
Generate new precision timing for Surah 90 using faster-whisper

Uses faster-whisper directly (which WhisperX wraps) to avoid pyannote VAD issues.
"""

import json
from pathlib import Path
from faster_whisper import WhisperModel

# Audio path
AUDIO_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/audio/abdul_basit/surah_090.mp3"
VERSES_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/data/verses_v4.json"
OUTPUT_PATH = Path(__file__).parent / "output/surah_90_new.json"

def run_alignment():
    print("=" * 60)
    print("Faster-Whisper Alignment - Surah 90")
    print("=" * 60)
    
    # Load model
    print("\n[1] Loading Whisper model (large-v3)...")
    model = WhisperModel("large-v3", device="cpu", compute_type="int8")
    
    # Transcribe with word timestamps
    print(f"\n[2] Transcribing: {AUDIO_PATH}")
    segments, info = model.transcribe(
        AUDIO_PATH,
        language="ar",
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    print(f"    Language: {info.language} (prob: {info.language_probability:.2f})")
    print(f"    Duration: {info.duration:.1f}s")
    
    # Extract word and character timing
    print("\n[3] Extracting letter timing...")
    letter_timing = []
    global_idx = 0
    all_segments = list(segments)
    
    print(f"    Segments: {len(all_segments)}")
    
    for segment in all_segments:
        if segment.words:
            for word in segment.words:
                word_text = word.word.strip()
                word_start = word.start
                word_end = word.end
                
                # Distribute timing across characters
                chars = list(word_text)
                if chars:
                    char_duration = (word_end - word_start) / len(chars)
                    for i, char in enumerate(chars):
                        char_start = word_start + (i * char_duration)
                        char_end = char_start + char_duration
                        letter_timing.append({
                            "char": char,
                            "start": round(char_start, 3),
                            "end": round(char_end, 3),
                            "idx": global_idx,
                            "word": word_text,
                            "source": "faster_whisper"
                        })
                        global_idx += 1
    
    print(f"    Total letters: {len(letter_timing)}")
    
    # Save output
    print(f"\n[4] Saving to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "surah": 90,
        "name": "Al-Balad",
        "source": "faster-whisper large-v3",
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration": round(info.duration, 1),
        "total_letters": len(letter_timing),
        "letters": letter_timing
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    # Save in MahQuranApp format
    mahquran_format = []
    for lt in letter_timing:
        mahquran_format.append({
            "char": lt["char"],
            "start": lt["start"],
            "end": lt["end"],
            "idx": lt["idx"]
        })
    
    mahquran_path = OUTPUT_PATH.parent / "letter_timing_90_new.json"
    with open(mahquran_path, 'w', encoding='utf-8') as f:
        json.dump(mahquran_format, f, ensure_ascii=False, indent=2)
    print(f"    Also saved: {mahquran_path}")
    
    print("\n" + "=" * 60)
    print("✓ Alignment complete!")
    print("=" * 60)
    
    # Show sample
    print("\nSample (first 10 letters):")
    for lt in letter_timing[:10]:
        print(f"  [{lt['char']}] {lt['start']:.3f}s - {lt['end']:.3f}s  ({lt['word']})")
    
    return letter_timing

if __name__ == "__main__":
    run_alignment()
