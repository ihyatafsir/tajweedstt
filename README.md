# TajweedSST — Quranic Letter-Level Alignment & Karaoke Video Renderer

> CTC Forced Alignment + Acoustic Physics Validation + Karaoke-Style Video Overlay for Quranic Recitation

## Overview

TajweedSST is a complete pipeline for producing **letter-level timing data** for Quranic recitation audio and rendering **karaoke-style overlay videos** with real-time per-letter highlighting synchronized to the reciter's voice.

### What It Does

1. **Aligns** — Uses wav2vec2 CTC forced alignment to produce per-letter timestamps from recitation audio
2. **Validates** — Applies Tajweed physics rules (Qalqalah, Ghunnah, Madd, Tafkheem) against acoustic data
3. **Renders** — Generates karaoke-style video with per-grapheme color animation, waveform visualization, and ayah-by-ayah display

### Output Example

The renderer produces an MP4 with:
- **Green glow** on the currently recited letter
- **Dimmed green** for past letters
- **White** for upcoming letters
- **Scrolling waveform** with playhead at bottom
- **Ayah number indicator** in gold
- **Semi-transparent overlay** with gradient fade over the background video

---

## Quick Start: Generate a Quran Karaoke Video

### Prerequisites

```bash
# System dependencies
sudo apt install ffmpeg
sudo apt install fonts-noto-extra  # NotoNaskhArabic font
# On Ubuntu/Debian also install libraqm for proper Arabic shaping:
sudo apt install libraqm-dev

# Python setup
cd /path/to/tajweedsst
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step-by-Step Process

#### Step 1: Prepare Your Source Material

You need three things:
1. **A video file** (background for the karaoke overlay, e.g. a nature scene or Islamic calligraphy)
2. **Audio of the recitation** (WAV format, can be extracted from a video)
3. **The Quran text** being recited (surah number + ayah range)

```bash
# If your audio is embedded in a video, extract it:
ffmpeg -i your_video.mp4 -vn -acodec pcm_s16le -ar 16000 -ac 1 audio.wav
```

#### Step 2: Generate Letter-Level Timing (CTC Alignment)

Edit `align_video.py` to set your config:

```python
# === CONFIG ===
AUDIO_PATH = Path("/path/to/your/audio.wav")
OUTPUT_PATH = Path("/path/to/output/letter_timing.json")
VERSES_PATH = Path("/path/to/verses_v4.json")  # Quran verse source
SURAH_NUM = 9       # Surah number
AYAH_START = 50     # First ayah
AYAH_END = 55       # Last ayah
```

Then run:

```bash
source venv/bin/activate
python align_video.py
```

This produces a JSON file with per-grapheme timing:
```json
[
  {
    "idx": 0,
    "char": "إِ",
    "ayah": 50,
    "wordIdx": 0,
    "start_s": 0.16,
    "end_s": 0.333,
    "duration_ms": 173.3
  }
]
```

#### Step 3: (Optional) Prepare Uthmani Display Text

For beautiful Uthmani-script display, create a JSON mapping of ayah numbers to their Uthmani text:

```json
{
  "50": "إِن تُصِبْكَ حَسَنَةٌۭ تَسُؤْهُمْ ۖ وَإِن تُصِبْكَ ...",
  "51": "قُل لَّن يُصِيبَنَآ إِلَّا مَا كَتَبَ ٱللَّهُ لَنَا ..."
}
```

> **Note:** The renderer automatically strips Quranic stop marks (ۖ, ۭ, ۚ, etc.) from the timing data to avoid rendering non-letter symbols.

#### Step 4: Render the Karaoke Video

Edit `render_quran_video.py` to set your paths:

```python
# === CONFIG ===
VIDEO_PATH = Path("/path/to/background_video.mp4")
AUDIO_PATH = Path("/path/to/audio.wav")
TIMING_PATH = Path("/path/to/letter_timing.json")
UTHMANI_PATH = Path("/path/to/uthmani_text.json")  # Optional
OUTPUT_PATH = Path("/path/to/output_karaoke.mp4")

WIDTH = 1024        # Video width
HEIGHT = 576        # Video height
FPS = 30            # Frames per second
OVERLAY_HEIGHT = 260  # Height of text overlay area
FONT_SIZE = 56      # Main Arabic text size
```

Then run:

```bash
source venv/bin/activate
python render_quran_video.py
```

**Processing time:** ~15-25 minutes for a 2-minute video on a modern CPU (3900+ frames at 30fps).

The renderer will:
1. Extract all frames from the background video
2. Render per-letter-colored Arabic text overlay on each frame
3. Assemble frames + audio into final MP4

---

## How It Works (Technical Deep-Dive)

### CTC Forced Alignment Pipeline

```
Audio (WAV) ──> wav2vec2 CTC ──> Word Timestamps ──> Character Expansion ──> Grapheme Matching
                                                                                     │
                                                                                     ▼
Quran Text ──> split_into_graphemes() ──> Grapheme List ──────────────────> Timing JSON
```

#### The Grapheme Matching Problem (CRITICAL)

This is the most important step. The timing entry count **must** exactly match the grapheme count used for rendering.

**The Problem:**
- Raw CTC produces ~651 individual characters (each letter and diacritic counted separately)
- Arabic rendering combines base letters with diacritics into ~345 graphemes
- If counts don't match → highlighting drifts!

**Solution:** We replicate the exact same `splitIntoGraphemes()` logic used by the renderer:

```python
DIACRITICS = set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ٰ',
                  'ۖ', 'ۗ', 'ۘ', 'ۙ', 'ۚ', 'ۛ', 'ۜ', 'ٔ', 'ٓ', 'ـ'])

def is_diacritic(ch):
    return ch in DIACRITICS or (0x064B <= ord(ch) <= 0x0652) or (0x0610 <= ord(ch) <= 0x061A)

def split_into_graphemes(text):
    graphemes = []
    current = ''
    for ch in text:
        if is_diacritic(ch) and current:
            current += ch  # Attach diacritic to base
        else:
            if current:
                graphemes.append(current)
            current = ch
    if current:
        graphemes.append(current)
    return graphemes
```

**Example:** The word `لَآ` splits into 2 graphemes:

| Grapheme | Characters | CTC entries consumed |
|----------|-----------|---------------------|
| `لَ` | ل + ـَ | 2 |
| `آ` | ا + ـٓ | 2 |

### Karaoke Video Renderer (v7 — RAQM Native Shaping)

The renderer uses **Pillow with RAQM/HarfBuzz** for proper Arabic text shaping:

```python
font = ImageFont.truetype(font_path, 56, layout_engine=ImageFont.Layout.RAQM)
```

**Per-frame rendering:**
1. Load background frame from video
2. Draw semi-transparent overlay with gradient fade
3. Find active grapheme at current timestamp (binary search)
4. Render current ayah with per-letter coloring:
   - **Active letter:** Bright green `(51, 255, 51)` with glow effect
   - **Past letters:** Dimmed green `(80, 160, 80)`
   - **Future letters:** White `(230, 230, 230)`
5. Draw scrolling waveform at bottom
6. Save as JPEG frame

**Multi-color word rendering** uses numpy-based compositing:
- Render word as white-on-transparent to get alpha mask
- Calculate grapheme widths proportionally
- Apply per-grapheme colors using column-based masking
- Composite result onto frame

### Tajweed Physics Validation

| Rule | Method | What it checks |
|------|--------|----------------|
| **Qalqalah** | RMS bounce detection | Dip followed by spike in amplitude |
| **Ghunnah** | Duration measurement | Nasal sound ≥ threshold duration |
| **Madd** | Duration ratio | Extended vowel vs base harakat |
| **Tafkheem** | Formant F2 analysis | Depressed F2 = heavy articulation |

---

## Reproducing for Any Quran Video

### Complete Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Get your recitation audio + background video         │
│    ffmpeg -i video.mp4 -vn -acodec pcm_s16le audio.wav  │
├─────────────────────────────────────────────────────────┤
│ 2. Identify which surah/ayahs are being recited         │
├─────────────────────────────────────────────────────────┤
│ 3. Run CTC alignment (align_video.py)                   │
│    → Produces letter_timing.json                        │
├─────────────────────────────────────────────────────────┤
│ 4. (Optional) Prepare Uthmani text JSON                 │
├─────────────────────────────────────────────────────────┤
│ 5. Run video renderer (render_quran_video.py)           │
│    → Produces karaoke MP4                               │
└─────────────────────────────────────────────────────────┘
```

### Customization Options

| Setting | File | Description |
|---------|------|-------------|
| `WIDTH`, `HEIGHT` | render_quran_video.py | Video resolution |
| `FPS` | render_quran_video.py | Frame rate (30 recommended) |
| `OVERLAY_HEIGHT` | render_quran_video.py | Text overlay area height |
| `FONT_SIZE` | render_quran_video.py | Main text font size |
| `COLOR_ACTIVE` | render_quran_video.py | Active letter color (default: green) |
| `COLOR_PAST` | render_quran_video.py | Past letter color |
| `COLOR_FUTURE` | render_quran_video.py | Future letter color |
| `SURAH_NUM` | align_video.py | Target surah |
| `AYAH_START/END` | align_video.py | Ayah range |

### Tips for Best Results

1. **Audio quality matters** — Use clean recitation audio (16kHz mono WAV works best for CTC)
2. **Match ayah boundaries** — Make sure your ayah range matches exactly what's in the audio
3. **Font choice** — NotoNaskhArabic provides the best results with RAQM shaping
4. **Verify timing** — Check the first/last few entries in the timing JSON to ensure timestamps look reasonable
5. **Adjust overlay height** — For ayahs with many words, increase `OVERLAY_HEIGHT` to prevent text cutoff

---

## Batch Processing (Full Quran)

For aligning all 114 surahs (for the MahQuranApp letter-level highlighting):

```bash
python batch_align_all.py
```

This uses the same CTC + physics pipeline but outputs timing files in the format expected by [MahQuranApp](https://github.com/ihyatafsir/MahQuranApp).

---

## Project Structure

```
tajweedsst/
├── render_quran_video.py      # 🎬 Karaoke video renderer (v7, RAQM)
├── align_video.py             # 🎯 CTC alignment for video audio
├── batch_align_all.py         # 📦 Batch all 114 surahs
├── ctc_align_90.py            # Surah 90 alignment (reference)
├── ctc_align_90_physics.py    # Surah 90 + physics validation
├── ctc_align_91.py            # Surah 91 with full physics
├── voice_server.py            # Real-time voice alignment server
├── requirements.txt
├── src/
│   ├── tajweed_parser.py      # Tajweed rule detection
│   ├── physics_validator.py   # Acoustic physics validation
│   ├── duration_model.py      # Duration calibration
│   ├── alignment_engine.py    # Core alignment engine
│   ├── pipeline.py            # Full pipeline orchestrator
│   ├── mfa_refiner.py         # Montreal Forced Aligner refinement
│   └── lisan_phonemes.json    # Phoneme mappings
├── tests/                     # Unit & integration tests
├── data/                      # Data files
├── models/                    # Model checkpoints
└── output/                    # Generated outputs
```

## Key Dependencies

| Package | Purpose |
|---------|---------|
| `Pillow` (with RAQM) | Arabic text rendering with HarfBuzz shaping |
| `numpy` | Fast per-grapheme color compositing |
| `torch` + `torchaudio` | wav2vec2 CTC model |
| `ctc-forced-aligner` | CTC forced alignment wrapper |
| `ffmpeg` (system) | Video frame extraction and assembly |
| `libraqm-dev` (system) | RAQM layout engine for Arabic |

## Reciter Support

| Reciter | Surahs | Status |
|---------|--------|--------|
| Abdul Basit | 114 | ✅ Complete |

## Lessons Learned

### Grapheme Count Must Match Exactly
The timing entry count **must** equal the grapheme count from `splitIntoGraphemes()`:
- 651 entries (raw CTC) → **failed** (each diacritic counted separately)
- 340 entries (wrong diacritics set) → **failed**
- **345 entries** (exact matching logic) → **success** ✅

### Quranic Stop Marks
Uthmani Quran text contains special stop/pause marks (ۖ, ۭ, ۚ, ۟) that render as circle-like symbols. These are **not** pronounced letters and must be stripped from the timing data to avoid phantom highlighting.

### RAQM vs Manual Reshaping
Early versions used `arabic_reshaper` + `python-bidi` for Arabic text. Switching to Pillow's native **RAQM layout engine** (backed by HarfBuzz + FriBidi) simplified the code significantly and produced more accurate glyph shaping.

### Font Sizing
Font size 56px with `OVERLAY_HEIGHT = 260` provides good readability on 1024×576 video. For 1080p output, scale proportionally.

## License

MIT
