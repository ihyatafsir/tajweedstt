# TajweedSST — Quranic Letter-Level Alignment & Tajweed Physics Engine

> CTC Forced Alignment + Acoustic Physics Validation for Quranic Recitation

## Overview

TajweedSST produces **letter-level timing data** for Quranic recitation audio. It combines **wav2vec2 CTC forced alignment** with **acoustic physics validation** (Tajweed rules) to generate timing files consumed by [MahQuranApp](https://github.com/ihyatafsir/MahQuranApp) for real-time letter-by-letter highlighting synchronized to audio.

---

## The Full Process (How We Built Surah 90)

This documents the exact process used to achieve accurate letter-level highlighting for Abdul Basit's recitation of Surah 90 (Al-Balad).

### Step 1: CTC Forced Alignment (wav2vec2)

We use [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) with wav2vec2 to get **word-level timestamps** from the audio.

```python
from ctc_forced_aligner import (
    load_audio, load_alignment_model,
    generate_emissions, preprocess_text,
    get_alignments, get_spans, postprocess_results,
)

# Load model + audio
alignment_model, alignment_tokenizer = load_alignment_model(DEVICE)
audio_waveform = load_audio("surah_090.mp3", ...)

# Generate CTC emissions
emissions, stride = generate_emissions(alignment_model, audio_waveform, batch_size=4)

# Preprocess Arabic text (romanize for CTC)
tokens_starred, text_starred = preprocess_text(text, romanize=True, language="ara")

# Get word alignments
segments, scores, blank_token = get_alignments(emissions, tokens_starred, alignment_tokenizer)
spans = get_spans(tokens_starred, segments, blank_token)
word_timestamps = postprocess_results(text_starred, spans, stride, scores)
```

**Output:** Word-level timestamps (start/end in seconds)

### Step 2: Character Expansion

CTC gives us word-level timing. We expand to individual character timing by distributing each word's duration across its characters:

```python
for wt in word_timestamps:
    word = wt['text']
    char_dur = (wt['end'] - wt['start']) / len(word)
    for i, char in enumerate(word):
        char_timings.append({
            'start': wt['start'] + i * char_dur,
            'end': wt['start'] + (i + 1) * char_dur,
        })
```

### Step 3: Grapheme Matching (CRITICAL)

This is the most important step. The timing data **must** have the exact same number of entries as the graphemes rendered by MahQuranApp.

**The Problem We Solved:**
- Raw CTC produces ~651 individual characters (each letter and diacritic separate)
- MahQuranApp's `splitIntoGraphemes()` combines base letters with diacritics into ~345 graphemes
- If counts don't match → highlighting drifts!

**App.tsx DIACRITICS Set (exact characters):**
```javascript
const DIACRITICS = new Set([
    'ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ',
    'ۖ', 'ۗ', 'ۘ', 'ۙ', 'ۚ', 'ۛ', 'ۜ', 'ٔ', 'ٓ', 'ـ'
]);
```

Plus Unicode ranges: `U+064B–U+0652` and `U+0610–U+061A`

**We replicate this exact logic in Python** to split the Quran verse text into graphemes, then map CTC character timings onto those graphemes:

```python
# Build target graphemes from verse text
grapheme_list = []
for verse in verses:
    for word in verse['text'].split():
        grapheme_list.extend(split_into_graphemes(word))  # Same logic as App.tsx

# Map CTC times to graphemes
timing = []
ctc_idx = 0
for gi, ginfo in enumerate(grapheme_list):
    grapheme = ginfo['char']  # e.g., 'لَ' (base + diacritic)
    start, end = None, None
    for _ in range(len(grapheme)):  # Consume 1 CTC entry per character
        if ctc_idx < len(char_timings):
            if start is None: start = char_timings[ctc_idx]['start']
            end = char_timings[ctc_idx]['end']
            ctc_idx += 1
    timing.append({'char': grapheme, 'start': start_ms, 'end': end_ms, ...})
```

**Example:** The word `لَآ` splits into 2 graphemes:
| Grapheme | Characters | CTC entries consumed |
|----------|-----------|---------------------|
| `لَ` | ل + ـَ | 2 |
| `آ` | ا + ـٓ | 2 |

### Step 4: Tajweed Parsing

Map each letter to its Tajweed rule using `TajweedParser`:

```python
from src.tajweed_parser import TajweedParser
parser = TajweedParser()
tags = parser.parse_text(verse_text)
# → Qalqalah, Ghunnah, Madd, Tafkheem, Idgham, etc.
```

### Step 5: Physics Validation

Validate Tajweed rules against acoustic physics using `PhysicsValidator`:

| Rule | Method | What it checks |
|------|--------|---------------|
| **Qalqalah** | RMS bounce detection | Dip followed by spike in amplitude |
| **Ghunnah** | Duration measurement | Nasal sound ≥ threshold duration |
| **Madd** | Duration ratio | Extended vowel vs base harakat |
| **Tafkheem** | Formant F2 analysis | Depressed F2 = heavy articulation |

```python
from src.physics_validator import PhysicsValidator
physics = PhysicsValidator(sample_rate=22050)

# Qalqalah: Check for RMS dip + spike pattern
result = physics.validate_qalqalah(audio, start_sec, end_sec)

# Madd: Check duration ratio against calibrated harakat
result = physics.validate_madd(audio, start_sec, end_sec, madd_count=2)
```

### Step 6: Convert to Milliseconds & Export

The app expects milliseconds (not seconds), plus `ayah`, `wordIdx`, and `duration` fields:

```python
timing.append({
    'idx': 0,
    'char': 'لَ',
    'ayah': 1,
    'start': 3360,      # milliseconds
    'end': 3410,         # milliseconds
    'duration': 50,      # ms
    'wordIdx': 0,
    'weight': 1.0
})
```

### Step 7: Deploy to MahQuranApp

Copy timing JSON to both `public/data/` and `dist/data/`:
```bash
cp letter_timing_90.json /path/to/MahQuranApp/public/data/abdul_basit/
cp letter_timing_90.json /path/to/MahQuranApp/dist/data/abdul_basit/
```

---

## Lessons Learned (Surah 90 Debugging)

### 1. Unit Mismatch (seconds vs milliseconds)
CTC outputs timestamps in **seconds** (3.36), but MahQuranApp expects **milliseconds** (3360). The app's `normalizeTimingToSeconds()` auto-detects this (`start > 100 → ms`) but the initial format must be consistent.

### 2. Grapheme Count Must Match Exactly
The timing entry count **must** equal the grapheme count from `splitIntoGraphemes()`. We went through several iterations:
- 651 entries (raw CTC) → **failed** (each diacritic counted separately)
- 340 entries (wrong diacritics set) → **failed** (7 missing chars)
- 353 entries (partial fix) → **failed** (off-by-one)
- **345 entries** (exact App.tsx logic) → **success!** ✅

### 3. Word-Calculated vs Letter-Level
Original Surah 90 had **word-calculated timing** (63% of letters at exactly 20ms = filler). Real letter-level alignment produces varied durations matching actual speech patterns.

| Metric | Word-Calculated | CTC Letter-Level |
|--------|----------------|-----------------|
| Letters at ≤20ms | 63% ❌ | 1% ✅ |
| Avg duration | 300ms | 161ms (split) / 309ms (merged) |
| Sync quality | Drift after ~10s | Accurate throughout |

---

## Quick Start

### Prerequisites
```bash
cd /path/to/tajweedsst
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio ctc-forced-aligner librosa
```

### Single Surah
```bash
python ctc_align_91.py  # Template with physics
```

### Batch All 114 Surahs
```bash
python batch_align_all.py
```

## Output Format

Each `letter_timing_XX.json` contains:
```json
{
  "idx": 0,
  "char": "لَ",
  "ayah": 1,
  "start": 3360,
  "end": 3410,
  "duration": 50,
  "wordIdx": 0,
  "weight": 1.0
}
```

## Project Structure

```
tajweedsst/
├── src/
│   ├── tajweed_parser.py       # Tajweed rule detection
│   ├── physics_validator.py    # Acoustic physics validation
│   └── duration_model.py       # Duration calibration model
├── tests/                      # 34 unit/integration tests
├── ctc_align_90.py             # Surah 90 alignment
├── ctc_align_91.py             # Surah 91 with physics
├── batch_align_all.py          # Batch all 114 surahs
└── README.md
```

## Reciter Support

| Reciter | Surahs | Status |
|---------|--------|--------|
| Abdul Basit | 114 | ✅ Complete |

## License

MIT
