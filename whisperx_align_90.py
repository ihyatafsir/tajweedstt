#!/usr/bin/env python3
"""
WhisperX Forced Alignment for Surah 90 (Al-Balad)
Uses wav2vec2 to FORCE align the known Quran text to the audio.
This gives perfect letter timing since we provide the exact text upfront.

Based on MahQuranApp/scripts/whisperx_forced_align.py
"""
import os
import json
import torch
import whisperx
from pathlib import Path

# Monkeypatch torch.load for PyTorch 2.6+ compatibility
try:
    from omegaconf import OmegaConf
    from omegaconf.listconfig import ListConfig
    from omegaconf.dictconfig import DictConfig
    from omegaconf.base import ContainerMetadata
    torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata])
    print("Added OmegaConf to torch safe globals.")
except ImportError:
    print("OmegaConf not found, using aggressive torch.load patch.")

original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False 
    return original_load(*args, **kwargs)
torch.load = safe_load

# Configuration
SURAH_NUM = 90
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
AUDIO_PATH = PROJECT_ROOT / "public/audio/abdul_basit/surah_090.mp3"
OUTPUT_DIR = PROJECT_ROOT / "public/data"
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
DEVICE = "cpu"  # Use CPU for compatibility

def get_surah_text():
    """Get Surah 90 text from verses_v4.json"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = ' '.join(v['text'] for v in data[str(SURAH_NUM)])
    return text

def main():
    print("=" * 60)
    print(f"WhisperX FORCED ALIGNMENT for Surah {SURAH_NUM} (Al-Balad)")
    print("Using known Quran text for direct wav2vec2 alignment")
    print("=" * 60)
    
    # 1. Check audio exists
    if not AUDIO_PATH.exists():
        print(f"ERROR: Audio not found at {AUDIO_PATH}")
        return
    
    # 2. Get Quran text
    quran_text = get_surah_text()
    print(f"\nQuran text ({len(quran_text)} chars):")
    print(quran_text[:100] + "...")
    
    # 3. Load Alignment Model (wav2vec2)
    print("\nLoading wav2vec2 alignment model (Arabic)...")
    model_a, metadata = whisperx.load_align_model(language_code="ar", device=DEVICE)
    print("Alignment model loaded.")
    
    # 4. Load Audio
    print("Loading audio...")
    audio = whisperx.load_audio(str(AUDIO_PATH))
    audio_duration = len(audio) / 16000  # Assuming 16kHz sample rate
    print(f"Audio duration: {audio_duration:.2f}s")
    
    # 5. Create "fake" segments from the known Quran text
    # WhisperX's align() function expects segments with 'text', 'start', 'end'
    # We provide the full Quran text as a single segment spanning the entire audio
    print("\nCreating forced alignment segment from Quran text...")
    segments = [{
        "text": quran_text,
        "start": 0.0,
        "end": audio_duration
    }]
    
    # 6. Force Align
    print("Performing FORCED ALIGNMENT with wav2vec2...")
    result = whisperx.align(
        segments, 
        model_a, 
        metadata, 
        audio, 
        DEVICE, 
        return_char_alignments=True
    )
    
    # 7. Extract character-level timing (SECONDS format for MahQuranApp)
    print("\nExtracting character timings...")
    output_timing = []
    idx = 0
    
    for seg in result.get("segments", []):
        if "chars" in seg:
            for ch in seg["chars"]:
                char = ch.get("char", "")
                start = ch.get("start", 0)
                end = ch.get("end", 0)
                
                # Skip spaces
                if char.isspace():
                    continue
                
                output_timing.append({
                    "char": char,
                    "start": round(start, 3),  # seconds
                    "end": round(end, 3),
                    "idx": idx
                })
                idx += 1
    
    print(f"Got {len(output_timing)} characters with timing")
    
    # 8. Save output
    output_path = OUTPUT_DIR / f"letter_timing_{SURAH_NUM}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_timing, f, ensure_ascii=False, indent=2)
        
    print(f"\nSaved to {output_path}")
    
    # Print first 20 for verification
    print("\n=== First 20 characters ===")
    for e in output_timing[:20]:
        dur_ms = (e['end'] - e['start']) * 1000
        print(f"  {e['idx']:3d}: '{e['char']}' @ {e['start']:.3f}s - {e['end']:.3f}s ({dur_ms:.0f}ms)")
    
    print("\n" + "=" * 60)
    print("✓ Forced alignment complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
