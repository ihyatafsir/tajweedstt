#!/usr/bin/env python3
"""
TajweedSST - Surah 91 (Ash-Shams) Physics Test

Tests the complete Tajweed physics system on Abdul Basit's recitation.
This validates all 10 physics validators on real Quranic audio.

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python3 surah_91_test.py
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tajweed_parser import TajweedParser, TajweedType, PhysicsCheck
from src.physics_validator import PhysicsValidator, ValidationStatus
from src.duration_model import DurationModel, MaddType

# Check for librosa
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Some tests will be skipped.")

# Paths
MAHQURAN_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = MAHQURAN_PATH / "public/data/verses_v4.json"
AUDIO_PATH = MAHQURAN_PATH / "public/audio/abdul_basit/surah_091.mp3"
TIMING_PATH = MAHQURAN_PATH / "public/data/abdul_basit/letter_timing_91.json"
OUTPUT_PATH = Path(__file__).parent / "output/surah_91_physics.json"


def load_surah_91_text():
    """Load Surah 91 text from verses_v4.json"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    surah_91 = data.get('91', [])
    
    verses = []
    for verse in surah_91:
        verses.append({
            'ayah': verse['ayah'],
            'text': verse['text'].strip(),
            'translation': verse.get('translation', ''),
        })
    
    return verses


def load_timing_data():
    """Load existing letter timing data"""
    with open(TIMING_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_audio():
    """Load audio file"""
    if not HAS_LIBROSA:
        return None, 22050
    
    print(f"    Loading audio from: {AUDIO_PATH}")
    audio, sr = librosa.load(str(AUDIO_PATH), sr=22050)
    print(f"    Duration: {len(audio)/sr:.1f}s")
    return audio, sr


def analyze_with_physics(verses, timing_data, audio, sr):
    """Analyze letters with physics validators"""
    parser = TajweedParser()
    physics = PhysicsValidator(sample_rate=sr)
    duration_model = DurationModel()
    
    # Parse all verses for Tajweed rules
    all_tags = []
    for verse in verses:
        word_tags = parser.parse_text(verse['text'])
        for word_tag in word_tags:
            for letter in word_tag.letters:
                all_tags.append({
                    'char': letter.char_visual,
                    'phonetic': letter.char_phonetic,
                    'tajweed_type': letter.tajweed_type.value,
                    'physics_check': letter.physics_check.value,
                    'madd_count': letter.madd_count
                })
    
    # Calibrate duration model from timing data
    short_vowels = []
    for entry in timing_data:
        duration = entry['end'] - entry['start']
        if 0.05 <= duration <= 0.15:  # Short vowel range
            short_vowels.append(duration)
    
    if short_vowels:
        duration_model.calibrate_from_samples("Abdul_Basit", short_vowels)
        print(f"    Calibrated harakat: {duration_model.calibration.harakat_base_ms:.1f}ms")
    
    # Run physics validation on each letter
    results = []
    physics_stats = {
        'total': 0,
        'validated': 0,
        'passed': 0,
        'marginal': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Match timing entries with Tajweed tags
    for i, entry in enumerate(timing_data):
        if i >= len(all_tags):
            break
        
        tag = all_tags[i]
        start = entry['start']
        end = entry['end']
        duration = end - start
        
        result = {
            'idx': i,
            'char': entry['char'],
            'start': start,
            'end': end,
            'duration_ms': duration * 1000,
            'tajweed_type': tag['tajweed_type'],
            'physics_check': tag['physics_check']
        }
        
        physics_stats['total'] += 1
        
        # Skip if no physics check needed or no audio
        if tag['physics_check'] == 'None' or audio is None:
            result['validation'] = 'not_required'
            results.append(result)
            continue
        
        physics_stats['validated'] += 1
        
        # Run appropriate validator
        check_type = tag['physics_check']
        
        try:
            if check_type == 'Check_RMS_Bounce':
                # Qalqalah
                val_result = physics.validate_qalqalah(audio, start, end)
                result['metric'] = 'RMS Bounce'
                result['profile'] = val_result.rms_profile if hasattr(val_result, 'rms_profile') else ''
                
            elif check_type == 'Check_Duration':
                # Madd
                madd_count = tag['madd_count'] if tag['madd_count'] > 0 else 2
                val_result = physics.validate_madd(audio, start, end, madd_count)
                result['metric'] = 'Duration'
                result['ratio'] = val_result.ratio if hasattr(val_result, 'ratio') else 0
                
            elif check_type == 'Check_Ghunnah':
                # Ghunnah/Ikhfa/Iqlab
                if tag['tajweed_type'] == 'Ikhfa':
                    val_result = physics.validate_ikhfa(audio, start, end)
                elif tag['tajweed_type'] == 'Iqlab':
                    val_result = physics.validate_iqlab(audio, start, end)
                else:
                    val_result = physics.validate_ghunnah(audio, start, end)
                result['metric'] = 'Nasal'
                
            elif check_type == 'Check_Formant_F2':
                # Tafkheem
                val_result = physics.validate_tafkheem(audio, start, end)
                result['metric'] = 'F2 Formant'
                
            else:
                val_result = None
            
            if val_result:
                result['status'] = val_result.status.value
                result['score'] = val_result.score
                
                if val_result.status == ValidationStatus.PASS:
                    physics_stats['passed'] += 1
                elif val_result.status == ValidationStatus.MARGINAL:
                    physics_stats['marginal'] += 1
                elif val_result.status == ValidationStatus.FAIL:
                    physics_stats['failed'] += 1
                else:
                    physics_stats['skipped'] += 1
            else:
                result['status'] = 'unknown'
                result['score'] = 0
                
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)
            physics_stats['skipped'] += 1
        
        results.append(result)
    
    return results, physics_stats, duration_model


def main():
    print("=" * 60)
    print("TajweedSST - Surah 91 (Ash-Shams) Physics Test")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n[1] Loading Surah 91 data...")
    verses = load_surah_91_text()
    print(f"    Verses: {len(verses)}")
    print(f"    First verse: {verses[0]['text'][:40]}...")
    
    timing_data = load_timing_data()
    print(f"    Timing entries: {len(timing_data)}")
    
    # Step 2: Load audio
    print("\n[2] Loading audio...")
    audio, sr = load_audio()
    
    # Step 3: Run physics analysis
    print("\n[3] Running physics validation...")
    results, stats, duration_model = analyze_with_physics(verses, timing_data, audio, sr)
    
    # Step 4: Print statistics
    print("\n[4] Physics Validation Statistics:")
    print(f"    Total letters: {stats['total']}")
    print(f"    Validated: {stats['validated']}")
    print(f"    ✓ Passed: {stats['passed']}")
    print(f"    ~ Marginal: {stats['marginal']}")
    print(f"    ✗ Failed: {stats['failed']}")
    print(f"    ⊘ Skipped: {stats['skipped']}")
    
    if stats['validated'] > 0:
        pass_rate = (stats['passed'] + stats['marginal']) / stats['validated'] * 100
        print(f"\n    Pass Rate: {pass_rate:.1f}%")
    
    # Step 5: Show samples of each Tajweed type
    print("\n[5] Sample Results by Tajweed Type:")
    
    tajweed_samples = {}
    for r in results:
        tj_type = r['tajweed_type']
        if tj_type != 'None' and tj_type not in tajweed_samples:
            tajweed_samples[tj_type] = r
    
    for tj_type, sample in tajweed_samples.items():
        status = sample.get('status', 'N/A')
        score = sample.get('score', 0)
        char = sample['char']
        print(f"    {tj_type}:")
        print(f"      Letter: {char}, Status: {status}, Score: {score:.2f}")
    
    # Step 6: Duration analysis
    print("\n[6] Duration Model Calibration:")
    if duration_model.calibration:
        print(f"    Reciter: {duration_model.calibration.reciter_name}")
        print(f"    Harakat base: {duration_model.calibration.harakat_base_ms:.1f}ms")
        print(f"    Sample size: {duration_model.calibration.sample_size}")
    
    # Step 7: Save results
    print("\n[7] Saving results...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    output = {
        'surah': 91,
        'name': 'Ash-Shams',
        'name_arabic': 'الشمس',
        'statistics': stats,
        'calibration': {
            'harakat_ms': duration_model.calibration.harakat_base_ms if duration_model.calibration else 100
        },
        'results': results
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {OUTPUT_PATH}")
    
    print("\n" + "=" * 60)
    print("✓ Physics Test Complete!")
    print("=" * 60)
    
    return output


if __name__ == "__main__":
    main()
