#!/usr/bin/env python3
"""
TajweedSST - Surah 90 Test

Test script to generate letter-level timing data for Surah Al-Balad (90)
and compare precision with existing timing in MahQuranApp.

Usage:
    cd /Documents/26apps/tajweedsst
    python3 surah_90_test.py
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.tajweed_parser import TajweedParser, TajweedType, PhysicsCheck

# Paths
MAHQURAN_PATH = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = MAHQURAN_PATH / "public/data/verses_v4.json"
AUDIO_PATH = MAHQURAN_PATH / "public/audio/abdul_basit/surah_090.mp3"
EXISTING_TIMING_PATH = MAHQURAN_PATH / "public/data/letter_timing_90.json"
OUTPUT_PATH = Path(__file__).parent / "output/surah_90_tajweed.json"


def load_surah_90_text():
    """Load Surah 90 text from verses_v4.json"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    surah_90 = data.get('90', [])
    
    verses = []
    for verse in surah_90:
        verses.append({
            'ayah': verse['ayah'],
            'text': verse['text'].strip(),
            'translation': verse.get('translation', ''),
            'words': [w['arabic'] for w in verse.get('words', [])]
        })
    
    return verses


def load_existing_timing():
    """Load existing timing data from MahQuranApp"""
    with open(EXISTING_TIMING_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def parse_with_tajweed(verses):
    """Parse all verses and generate Tajweed tags"""
    parser = TajweedParser()
    
    all_results = []
    
    for verse in verses:
        text = verse['text']
        word_tags = parser.parse_text(text)
        
        verse_result = {
            'ayah': verse['ayah'],
            'text': text,
            'translation': verse['translation'],
            'words': []
        }
        
        for word_tag in word_tags:
            word_result = {
                'word_text': word_tag.word_text,
                'phonetic': word_tag.phonetic_stream,
                'letters': []
            }
            
            for letter in word_tag.letters:
                letter_result = {
                    'char': letter.char_visual,
                    'phonetic': letter.char_phonetic,
                    'position': letter.position,
                    'tajweed_type': letter.tajweed_type.value,
                    'physics_check': letter.physics_check.value,
                    'is_silent': letter.is_silent,
                    'madd_count': letter.madd_count
                }
                word_result['letters'].append(letter_result)
            
            verse_result['words'].append(word_result)
        
        all_results.append(verse_result)
    
    return all_results


def analyze_tajweed_distribution(results):
    """Analyze distribution of Tajweed rules in Surah 90"""
    tajweed_counts = {}
    physics_counts = {}
    
    for verse in results:
        for word in verse['words']:
            for letter in word['letters']:
                tajweed_type = letter['tajweed_type']
                physics_check = letter['physics_check']
                
                tajweed_counts[tajweed_type] = tajweed_counts.get(tajweed_type, 0) + 1
                physics_counts[physics_check] = physics_counts.get(physics_check, 0) + 1
    
    return tajweed_counts, physics_counts


def convert_to_mahquran_format(results, existing_timing):
    """
    Convert TajweedSST output to MahQuranApp timing format.
    Uses existing timing as base and adds Tajweed annotations.
    """
    output = []
    char_idx = 0
    
    # Build a flat list of all characters with Tajweed info
    tajweed_map = {}
    global_idx = 0
    
    for verse in results:
        for word in verse['words']:
            for letter in word['letters']:
                tajweed_map[global_idx] = {
                    'tajweed_type': letter['tajweed_type'],
                    'physics_check': letter['physics_check'],
                    'phonetic': letter['phonetic'],
                    'madd_count': letter['madd_count']
                }
                global_idx += 1
    
    # Merge with existing timing
    for i, timing_entry in enumerate(existing_timing):
        entry = timing_entry.copy()
        
        # Add Tajweed info if available
        if i in tajweed_map:
            entry['tajweed_type'] = tajweed_map[i]['tajweed_type']
            entry['physics_check'] = tajweed_map[i]['physics_check']
            entry['phonetic'] = tajweed_map[i]['phonetic']
            if tajweed_map[i]['madd_count'] > 0:
                entry['madd_count'] = tajweed_map[i]['madd_count']
        
        output.append(entry)
    
    return output


def main():
    print("=" * 60)
    print("TajweedSST - Surah 90 (Al-Balad) Test")
    print("=" * 60)
    
    # Step 1: Load Surah 90 text
    print("\n[1] Loading Surah 90 text...")
    verses = load_surah_90_text()
    print(f"    Loaded {len(verses)} verses")
    print(f"    Verse 1: {verses[0]['text'][:50]}...")
    
    # Step 2: Parse with Tajweed
    print("\n[2] Parsing with Tajweed rules...")
    results = parse_with_tajweed(verses)
    
    # Step 3: Analyze distribution
    print("\n[3] Tajweed Analysis:")
    tajweed_counts, physics_counts = analyze_tajweed_distribution(results)
    
    print("\n    Tajweed Rules Found:")
    for rule, count in sorted(tajweed_counts.items(), key=lambda x: -x[1]):
        if rule != "None":
            print(f"      • {rule}: {count}")
    
    print("\n    Physics Checks Required:")
    for check, count in sorted(physics_counts.items(), key=lambda x: -x[1]):
        if check != "None":
            print(f"      • {check}: {count}")
    
    # Step 4: Load existing timing
    print("\n[4] Loading existing timing data...")
    existing_timing = load_existing_timing()
    print(f"    Found {len(existing_timing)} timing entries")
    print(f"    First entry: {existing_timing[0]}")
    
    # Step 5: Convert and merge
    print("\n[5] Merging Tajweed with timing...")
    merged = convert_to_mahquran_format(results, existing_timing)
    
    # Count enhanced entries
    enhanced = sum(1 for e in merged if e.get('tajweed_type') and e['tajweed_type'] != 'None')
    print(f"    Enhanced entries with Tajweed: {enhanced}")
    
    # Step 6: Save output
    print("\n[6] Saving output...")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full Tajweed analysis
    full_output = {
        'surah': 90,
        'name': 'Al-Balad',
        'name_arabic': 'البلد',
        'total_verses': len(verses),
        'tajweed_summary': tajweed_counts,
        'physics_checks': physics_counts,
        'verses': results
    }
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(full_output, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {OUTPUT_PATH}")
    
    # Save merged timing (compatible with MahQuranApp)
    merged_path = OUTPUT_PATH.parent / "letter_timing_90_tajweed.json"
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"    Saved: {merged_path}")
    
    # Step 7: Show sample
    print("\n[7] Sample Output (Verse 1, first 3 words):")
    for word in results[0]['words'][:3]:
        print(f"\n    Word: {word['word_text']}")
        print(f"    Phonetic: {word['phonetic']}")
        for letter in word['letters']:
            if letter['tajweed_type'] != 'None':
                print(f"      [{letter['char']}] → {letter['tajweed_type']} ({letter['physics_check']})")
    
    print("\n" + "=" * 60)
    print("✓ Test Complete!")
    print("=" * 60)
    
    return full_output


if __name__ == "__main__":
    main()
