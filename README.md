# TajweedSST — Quranic Letter-Level Alignment & Tajweed Physics Engine

> CTC Forced Alignment + Acoustic Physics Validation for Quranic Recitation

## Overview

TajweedSST is a Python pipeline that produces **letter-level timing data** for Quranic recitation audio. It combines **wav2vec2 CTC forced alignment** with **acoustic physics validation** (Tajweed rules) to generate timing files consumed by [MahQuranApp](https://github.com/ihyatafsir/MahQuranApp) for real-time letter highlighting.

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TajweedSST Pipeline                       │
│                                                             │
│  1. CTC Forced Alignment (wav2vec2)                         │
│     └─ Word-level timestamps from audio                     │
│                                                             │
│  2. Character Expansion                                     │
│     └─ Word timestamps → individual character timing        │
│                                                             │
│  3. Grapheme Matching                                       │
│     └─ Merge base + diacritics to match App.tsx rendering   │
│                                                             │
│  4. Tajweed Parsing                                         │
│     └─ Map letters to Tajweed rules (Qalqalah, Ghunnah..)  │
│                                                             │
│  5. Physics Validation                                      │
│     └─ RMS bounce, duration, formant analysis               │
│                                                             │
│  6. Export to MahQuranApp format                             │
│     └─ JSON with idx, char, ayah, start(ms), end, wordIdx   │
└─────────────────────────────────────────────────────────────┘
```

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
# Align Surah 90 (Al-Balad) for Abdul Basit
python ctc_align_91.py  # Template script
```

### Batch All Surahs

```bash
# Process all 114 surahs for Abdul Basit
python batch_align_all.py
```

## Output Format

Each `letter_timing_XX.json` contains an array of timing entries:

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

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `idx` | int | Sequential letter index |
| `char` | string | Arabic grapheme (base + diacritics) |
| `ayah` | int | Verse number (1-indexed) |
| `start` | int | Start time in milliseconds |
| `end` | int | End time in milliseconds |
| `duration` | int | Duration in milliseconds |
| `wordIdx` | int | Word index within the surah |
| `weight` | float | Confidence weight |

## Critical: Grapheme Matching

The timing data **must** match the grapheme count produced by MahQuranApp's `splitIntoGraphemes()` function. This function combines base Arabic letters with their following diacritics:

**App.tsx Diacritics Set:**
```
ً ٌ ٍ َ ُ ِ ّ ْ ٰ ۖ ۗ ۘ ۙ ۚ ۛ ۜ ٔ ٓ ـ
```

Plus Unicode ranges: `0x064B–0x0652` and `0x0610–0x061A`

**Example:** The word `لَآ` splits into 2 graphemes: `['لَ', 'آ']`

If the timing count doesn't match the grapheme count, highlighting will drift!

## Physics Validation

TajweedSST validates timing against acoustic physics:

| Rule | Check | Method |
|------|-------|--------|
| Qalqalah | RMS dip + spike | Envelope analysis |
| Ghunnah | Nasal duration | Duration measurement |
| Madd | Extended vowel | Duration ratio |
| Tafkheem | Heavy articulation | Formant F2 analysis |

## Project Structure

```
tajweedsst/
├── src/
│   ├── tajweed_parser.py     # Tajweed rule detection
│   ├── physics_validator.py  # Acoustic validation
│   └── duration_model.py     # Duration calibration
├── tests/                    # 34 unit/integration tests
├── ctc_align_90.py          # Single surah alignment
├── ctc_align_91.py          # Template with physics
├── batch_align_all.py       # Batch all surahs
└── README.md
```

## Reciter Support

Currently supported:
- **Abdul Basit** (114 surahs)

## License

MIT
