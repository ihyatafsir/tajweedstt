#!/usr/bin/env python3
"""
Grapheme-Aligned Timing Generator for Surah 91

This script:
1. Reads verse text from verses_v4.json and extracts graphemes (exactly as MahQuranApp does)
2. Reads the original timing and maps it to the grapheme count
3. Outputs timing with exactly the right number of entries

The key is: timing entries must match the grapheme count from verse.words[].arabic
"""
import json
from pathlib import Path

# Config
SURAH = 91
PROJECT_ROOT = Path("/home/absolut7/Documents/26apps/MahQuranApp")
VERSES_PATH = PROJECT_ROOT / "public/data/verses_v4.json"
TIMING_PATH = PROJECT_ROOT / "public/data/abdul_basit_original/letter_timing_91.json"
OUTPUT_PATH = PROJECT_ROOT / "public/data/abdul_basit/letter_timing_91_aligned.json"

# Arabic diacritics (same as MahQuranApp App.tsx)
DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')


def split_graphemes(text: str) -> list[str]:
    """Split Arabic text into graphemes (base letter + following diacritics)
    This matches the splitIntoGraphemes function in MahQuranApp"""
    graphemes = []
    current = ''
    
    for ch in text:
        is_diacritic = (ch in DIACRITICS or 
                        (0x064B <= ord(ch) <= 0x0652) or 
                        (0x0610 <= ord(ch) <= 0x061A))
        
        if ch == ' ':
            if current:
                graphemes.append(current)
                current = ''
        elif is_diacritic and current:
            current += ch
        else:
            if current:
                graphemes.append(current)
            current = ch
    
    if current:
        graphemes.append(current)
    
    return graphemes


def get_all_graphemes(surah_num: int) -> list[dict]:
    """Extract all graphemes from verse text, exactly as MahQuranApp renders them"""
    with open(VERSES_PATH, 'r', encoding='utf-8') as f:
        verses = json.load(f).get(str(surah_num), [])
    
    all_graphemes = []
    word_idx = 0
    
    for verse in verses:
        ayah = verse.get('ayah', 0)
        words = verse.get('words', [])
        
        for word in words:
            arabic = word.get('arabic', '')
            graphemes = split_graphemes(arabic)
            
            for g in graphemes:
                all_graphemes.append({
                    'char': g,
                    'ayah': ayah,
                    'wordIdx': word_idx
                })
            
            word_idx += 1
    
    return all_graphemes


def strip_diacritics(text: str) -> str:
    """Remove diacritics from Arabic text"""
    return ''.join(ch for ch in text if ch not in DIACRITICS and not (0x064B <= ord(ch) <= 0x0652))


def is_standalone_diacritic(char: str) -> bool:
    """Check if char is a standalone diacritic"""
    if len(char) != 1:
        return False
    return char in DIACRITICS or (0x064B <= ord(char) <= 0x0652)


def distribute_timing(graphemes: list[dict], original_timing: list[dict]) -> list[dict]:
    """Map original timing to graphemes by matching base letters, skipping diacritics"""
    if not original_timing:
        return []
    
    # First, filter out standalone diacritics from original timing
    # and merge their duration into the previous letter
    filtered_timing = []
    for entry in original_timing:
        char = entry['char']
        if is_standalone_diacritic(char):
            # Merge duration into previous entry
            if filtered_timing:
                filtered_timing[-1]['end'] = entry['end']
                filtered_timing[-1]['duration'] = filtered_timing[-1]['end'] - filtered_timing[-1]['start']
        else:
            filtered_timing.append(dict(entry))  # Copy
    
    print(f"    (Filtered timing: {len(filtered_timing)} base letters)")
    
    aligned_timing = []
    orig_idx = 0
    
    for i, g in enumerate(graphemes):
        grapheme_char = g['char']
        base_letter = strip_diacritics(grapheme_char)
        
        # Try to find matching original timing entry by base letter
        matched = None
        search_start = max(0, orig_idx - 2)
        search_end = min(len(filtered_timing), orig_idx + 10)  # Search wider
        
        for j in range(search_start, search_end):
            orig_char = filtered_timing[j]['char']
            orig_base = strip_diacritics(orig_char)
            if orig_base == base_letter or orig_char in grapheme_char or base_letter in orig_char:
                matched = filtered_timing[j]
                orig_idx = j + 1
                break
        
        if not matched and orig_idx < len(filtered_timing):
            # Fallback: use next available timing
            matched = filtered_timing[orig_idx]
            orig_idx += 1
        
        if matched:
            aligned_timing.append({
                'idx': i,
                'char': grapheme_char,
                'ayah': g['ayah'],
                'start': matched['start'],
                'end': matched['end'],
                'duration': matched.get('duration', matched['end'] - matched['start']),
                'wordIdx': g['wordIdx'],
                'weight': matched.get('weight', 1.0)
            })
        else:
            # Last resort: estimate from previous
            if aligned_timing:
                prev = aligned_timing[-1]
                aligned_timing.append({
                    'idx': i,
                    'char': grapheme_char,
                    'ayah': g['ayah'],
                    'start': prev['end'],
                    'end': prev['end'] + 100,
                    'duration': 100,
                    'wordIdx': g['wordIdx'],
                    'weight': 1.0
                })
    
    return aligned_timing


def main():
    print("=" * 60)
    print(f"Grapheme-Aligned Timing Generator: Surah {SURAH}")
    print("=" * 60)
    
    # Get graphemes from verse text
    graphemes = get_all_graphemes(SURAH)
    print(f"\n[1] Graphemes from verse text: {len(graphemes)}")
    
    # Load original timing
    with open(TIMING_PATH, 'r', encoding='utf-8') as f:
        original_timing = json.load(f)
    print(f"[2] Original timing entries: {len(original_timing)}")
    
    # Distribute timing to graphemes
    aligned_timing = distribute_timing(graphemes, original_timing)
    print(f"[3] Aligned timing entries: {len(aligned_timing)}")
    
    # Save
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(aligned_timing, f, ensure_ascii=False, indent=2)
    print(f"\n[4] Saved: {OUTPUT_PATH}")
    
    # Show sample
    print("\n=== First 10 graphemes ===")
    for t in aligned_timing[:10]:
        print(f"  {t['idx']:3d}: '{t['char']}' @ {t['start']}-{t['end']}ms (ayah={t['ayah']})")
    
    print("\n" + "=" * 60)
    print("✓ Done! Copy to letter_timing_91.json to test")
    print("=" * 60)


if __name__ == "__main__":
    main()
