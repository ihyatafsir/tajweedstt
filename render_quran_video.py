#!/usr/bin/env python3
"""
Quran Letter-Level Video Renderer v7 (RAQM Native Shaping)
Per-grapheme coloring with native harfbuzz/raqm Arabic shaping.
Static text layout (no scrolling), bigger font, Uthmani Quran text.
Renders overlay frames to JPEG files, then assembles with ffmpeg.

Usage:
    cd /Documents/26apps/tajweedsst
    source venv/bin/activate
    python render_quran_video.py
"""
import json
import subprocess
import wave
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# === CONFIG ===
VIDEO_PATH = Path("/home/absolut7/Downloads/twitter_video.mp4")
AUDIO_PATH = Path("/home/absolut7/Downloads/twitter_audio.wav")
TIMING_PATH = Path("/home/absolut7/Downloads/video_letter_timing.json")
UTHMANI_PATH = Path("/home/absolut7/Downloads/uthmani_9_50_55.json")
OUTPUT_PATH = Path("/home/absolut7/Downloads/quran_tilawah_overlay.mp4")
FRAMES_DIR = Path("/tmp/quran_frames_v7")

WIDTH = 1024
HEIGHT = 576
FPS = 30
OVERLAY_HEIGHT = 260
WAVEFORM_HEIGHT = 26

# Colors (RGBA)
COLOR_FUTURE = (230, 230, 230, 255)
COLOR_ACTIVE = (51, 255, 51, 255)
COLOR_ACTIVE_GLOW = (51, 255, 51, 80)
COLOR_PAST = (80, 160, 80, 255)
AYAH_COLOR = (255, 215, 0, 200)

FONT_SIZE = 56
FONT_SIZE_SM = 36


def load_waveform(audio_path, duration_s, rate=80):
    with wave.open(str(audio_path), 'rb') as wf:
        raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    total = int(duration_s * rate)
    chunk = max(1, len(samples) // total)
    return np.array([np.max(np.abs(samples[i:i+chunk])) for i in range(0, len(samples), chunk)])[:total]


def find_font():
    # Use Noto Naskh Arabic for proper rendering with harfbuzz/raqm
    for p in ["/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
              "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
              "/home/absolut7/Documents/26apps/MahQuranApp/public/fonts/AmiriQuran.ttf"]:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No Arabic font found!")


DIACRITICS = set('ًٌٍَُِّْٰۖۗۘۙۚۛۜٔٓـ')

def is_diacritic(ch):
    cp = ord(ch)
    return (ch in DIACRITICS or (0x064B <= cp <= 0x0652) or (0x0610 <= cp <= 0x061A))


def load_uthmani_text():
    """Load Uthmani Quran text with full diacritics."""
    if UTHMANI_PATH.exists():
        with open(UTHMANI_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


# Quranic stop/pause marks to strip from source data
QURAN_STOP_MARKS = set('\u06D6\u06D7\u06D8\u06D9\u06DA\u06DB\u06DC\u06DD\u06DE\u06DF\u06E0\u06E1\u06E2\u06E3\u06E4\u06E5\u06E6\u06E7\u06E8\u06E9\u06EA\u06EB\u06EC\u06ED')

def clean_timing_data(timing):
    """Strip Quranic stop marks from grapheme char fields in timing data."""
    for g in timing:
        g['char'] = ''.join(c for c in g['char'] if c not in QURAN_STOP_MARKS)
    return timing


def precompute_ayahs(timing, uthmani_text=None):
    ayahs = {}
    for t in timing:
        ayahs.setdefault(t['ayah'], []).append(t)

    result = []
    for ayah_num in sorted(ayahs.keys()):
        graphemes = ayahs[ayah_num]
        words = []
        current_chars = []
        current_widx = None

        for g in graphemes:
            if current_widx is not None and g['wordIdx'] != current_widx:
                raw = ''.join(c['char'] for c in current_chars)
                words.append({'raw': raw, 'display': raw, 'graphemes': current_chars})
                current_chars = []
            current_chars.append(g)
            current_widx = g['wordIdx']

        if current_chars:
            raw = ''.join(c['char'] for c in current_chars)
            words.append({'raw': raw, 'display': raw, 'graphemes': current_chars})

        result.append({
            'ayah': ayah_num, 'words': words,
            'start_s': graphemes[0]['start_s'], 'end_s': graphemes[-1]['end_s'],
        })
    return result


def render_colored_word(draw, word, active_idx, x, y, font, is_current):
    """Render word with per-grapheme coloring using native RAQM shaping."""
    graphemes = word['graphemes']
    display_text = word['display']
    n = len(graphemes)

    bbox = font.getbbox(display_text, direction='rtl')
    word_w = bbox[2] - bbox[0]

    if not is_current:
        color = tuple(int(c * 0.45) for c in COLOR_PAST[:3])
        draw.text((x, y), display_text, fill=color, font=font, direction='rtl')
        return word_w

    # Determine grapheme colors
    colors = []
    for g in graphemes:
        if g['idx'] == active_idx:
            colors.append(COLOR_ACTIVE)
        elif g['idx'] < active_idx:
            colors.append(COLOR_PAST)
        else:
            colors.append(COLOR_FUTURE)

    # Fast path: all same color
    if all(c == colors[0] for c in colors):
        if colors[0] == COLOR_ACTIVE:
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    draw.text((x + dx, y + dy), display_text, fill=COLOR_ACTIVE_GLOW,
                              font=font, direction='rtl')
        draw.text((x, y), display_text, fill=colors[0][:3], font=font, direction='rtl')
        return word_w

    # Multi-color: use numpy for fast compositing
    pad = 10
    tmp_w = word_w + pad * 2
    tmp_h = FONT_SIZE + 40  # Extra space for diacritics

    # Render word as white-on-transparent to get alpha mask
    word_img = Image.new('RGBA', (tmp_w, tmp_h), (0, 0, 0, 0))
    ImageDraw.Draw(word_img).text((pad, 10), display_text,
                                  fill=(255, 255, 255, 255), font=font, direction='rtl')

    # Convert to numpy for fast operations
    word_arr = np.array(word_img)
    alpha_mask = word_arr[:, :, 3]  # Alpha channel

    # Create color strip
    result_arr = np.zeros_like(word_arr)

    # Calculate grapheme widths (proportional based on base char count)
    base_counts = []
    for g in graphemes:
        base = sum(1 for ch in g['char'] if not is_diacritic(ch))
        base_counts.append(max(1, base))
    total_bases = sum(base_counts)

    # Build color map column by column
    # RAQM renders RTL natively, so grapheme 0 is on the RIGHT side
    cum_x = pad
    for i in range(n):
        g_w = int(word_w * base_counts[i] / total_bases)
        if i == n - 1:
            g_w = pad + word_w - cum_x  # Last one fills remaining

        # With RAQM, the text is already in display order (RTL)
        # grapheme 0 (first in Arabic reading) -> right side of display
        rtl_i = n - 1 - i
        color = colors[rtl_i]

        x_start = cum_x
        x_end = min(cum_x + g_w, tmp_w)

        # Apply glow for active
        if color == COLOR_ACTIVE:
            gx_s = max(0, x_start - 1)
            gx_e = min(tmp_w, x_end + 1)
            glow_mask = alpha_mask[:, gx_s:gx_e] > 0
            for c_idx in range(3):
                result_arr[:, gx_s:gx_e, c_idx] = np.where(
                    glow_mask, COLOR_ACTIVE_GLOW[c_idx], result_arr[:, gx_s:gx_e, c_idx])
            result_arr[:, gx_s:gx_e, 3] = np.where(
                glow_mask, COLOR_ACTIVE_GLOW[3], result_arr[:, gx_s:gx_e, 3])

        # Main color
        col_mask = alpha_mask[:, x_start:x_end] > 0
        for c_idx in range(3):
            result_arr[:, x_start:x_end, c_idx] = np.where(
                col_mask, color[c_idx], result_arr[:, x_start:x_end, c_idx])
        result_arr[:, x_start:x_end, 3] = np.where(
            col_mask, alpha_mask[:, x_start:x_end], result_arr[:, x_start:x_end, 3])

        cum_x += g_w

    # Paste result onto frame
    result_img = Image.fromarray(result_arr)
    frame_img = draw._image
    frame_img.paste(result_img, (x - pad, y - 10), result_img)

    return word_w


def find_active_idx(timing, t):
    if not timing or t < timing[0]['start_s']:
        return -1
    if t >= timing[-1]['start_s']:
        return timing[-1]['idx']
    lo, hi = 0, len(timing) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if timing[mid]['start_s'] <= t:
            lo = mid
        else:
            hi = mid - 1
    return timing[lo]['idx']


def wrap_words_to_lines(words, font, max_width, space_w):
    """Break words into lines that fit within max_width. Returns list of (words, widths) tuples."""
    lines = []
    current_words = []
    current_widths = []
    current_w = 0

    for w in words:
        bbox = font.getbbox(w['display'], direction='rtl')
        ww = bbox[2] - bbox[0]
        needed = ww + (space_w if current_words else 0)

        if current_words and current_w + needed > max_width:
            lines.append((current_words, current_widths))
            current_words = []
            current_widths = []
            current_w = 0

        current_words.append(w)
        current_widths.append(ww)
        current_w += needed

    if current_words:
        lines.append((current_words, current_widths))

    return lines


def render_frame(frame_img, ayah_groups, timing, current_time, waveform, wf_rate, font, font_sm):
    draw = ImageDraw.Draw(frame_img, 'RGBA')

    overlay_y = HEIGHT - OVERLAY_HEIGHT
    draw.rectangle([(0, overlay_y), (WIDTH, HEIGHT)], fill=(0, 0, 0, 255))
    for i in range(50):
        y = overlay_y - 50 + i
        if y < 0: continue
        draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, int(255 * i / 50)))

    # Waveform
    wf_y = overlay_y + 2
    wf_mid = wf_y + WAVEFORM_HEIGHT // 2
    playback_pos = int(current_time * wf_rate)
    playhead_x = WIDTH - 80

    for x in range(10, WIDTH - 10):
        si = playback_pos - (playhead_x - x)
        if 0 <= si < len(waveform):
            amp = waveform[si]
            h = max(1, int(amp * WAVEFORM_HEIGHT * 0.85))
            if x < playhead_x:
                frac = max(0.1, (x - 10) / max(1, playhead_x - 10))
                c = (51, 200, 51, int(60 + 100 * frac))
            else:
                frac = max(0, 1 - (x - playhead_x) / 100)
                c = (51, 255, 51, int(50 + 150 * frac))
            draw.line([(x, wf_mid - h), (x, wf_mid + h)], fill=c, width=1)

    draw.line([(playhead_x, wf_y), (playhead_x, wf_y + WAVEFORM_HEIGHT)],
              fill=(255, 255, 255, 200), width=2)

    # Text - STATIC layout (no scrolling)
    active_idx = find_active_idx(timing, current_time)
    if active_idx < 0: active_idx = 0
    active_ayah = timing[min(active_idx, len(timing)-1)]['ayah']

    ayah_nums = [ag['ayah'] for ag in ayah_groups]
    try:
        cur_pos = ayah_nums.index(active_ayah)
    except ValueError:
        cur_pos = 0

    # Only show current ayah (to maximize text size)
    ag = ayah_groups[cur_pos]
    space_w = 10
    max_text_width = WIDTH - 80  # margins

    # Wrap words into lines
    lines = wrap_words_to_lines(ag['words'], font, max_text_width, space_w)

    # Calculate vertical positioning to center text in overlay area
    line_height = FONT_SIZE + 14
    total_text_height = len(lines) * line_height
    text_start_y = overlay_y + WAVEFORM_HEIGHT + 8 + max(0, (OVERLAY_HEIGHT - WAVEFORM_HEIGHT - 16 - total_text_height) // 2)

    # Render each line (static position, only colors change)
    for line_idx, (line_words, line_widths) in enumerate(lines):
        total_line_w = sum(line_widths) + space_w * (len(line_widths) - 1)
        x = (WIDTH + total_line_w) // 2  # Start from right for RTL
        line_y = text_start_y + line_idx * line_height

        for w, ww in zip(line_words, line_widths):
            x -= ww
            render_colored_word(draw, w, active_idx, x, line_y, font, True)
            x -= space_w

    # Ayah number indicator
    ayah_str = f"\ufd3f{ag['ayah']}\ufd3e"
    draw.text((WIDTH - 60, text_start_y + 2), ayah_str, fill=AYAH_COLOR, font=font_sm)

    return frame_img


def main():
    print("=" * 60)
    print("Quran Letter-Level Video Renderer v7 (RAQM Native Shaping)")
    print("=" * 60)

    # 1. Load
    print("\n[1] Loading...", flush=True)
    with open(TIMING_PATH, 'r', encoding='utf-8') as f:
        timing = json.load(f)
    timing = clean_timing_data(timing)
    duration_s = timing[-1]['end_s']
    print(f"    {len(timing)} graphemes, {duration_s:.1f}s", flush=True)

    font_path = find_font()
    print(f"    Font: {font_path}", flush=True)
    font = ImageFont.truetype(font_path, FONT_SIZE, layout_engine=ImageFont.Layout.RAQM)
    font_sm = ImageFont.truetype(font_path, FONT_SIZE_SM, layout_engine=ImageFont.Layout.RAQM)

    uthmani_text = load_uthmani_text()
    if uthmani_text:
        print(f"    Loaded Uthmani text for {len(uthmani_text)} ayahs", flush=True)
    ayah_groups = precompute_ayahs(timing, uthmani_text)
    wf_rate = 80
    waveform = load_waveform(AUDIO_PATH, duration_s, wf_rate)
    print(f"    {len(ayah_groups)} ayahs, {len(waveform)} waveform samples", flush=True)

    # 2. Extract frames
    total_frames = int(duration_s * FPS)
    print(f"\n[2] Extracting {total_frames} frames from video...", flush=True)
    if FRAMES_DIR.exists():
        shutil.rmtree(FRAMES_DIR)
    FRAMES_DIR.mkdir(parents=True)

    # Use ffmpeg to extract all frames as JPEG
    subprocess.run([
        'ffmpeg', '-y', '-i', str(VIDEO_PATH),
        '-vf', f'fps={FPS}',
        '-q:v', '3',
        str(FRAMES_DIR / 'frame_%05d.jpg')
    ], capture_output=True, check=True)

    frame_files = sorted(FRAMES_DIR.glob('frame_*.jpg'))
    print(f"    Extracted {len(frame_files)} frames", flush=True)

    # 3. Render overlays
    out_dir = FRAMES_DIR / 'out'
    out_dir.mkdir()
    print(f"\n[3] Rendering overlays on {len(frame_files)} frames...", flush=True)

    for i, frame_path in enumerate(frame_files[:total_frames]):
        frame_img = Image.open(frame_path).convert('RGBA')
        # Resize if needed
        if frame_img.size != (WIDTH, HEIGHT):
            frame_img = frame_img.resize((WIDTH, HEIGHT), Image.LANCZOS)

        t = i / FPS
        frame_img = render_frame(frame_img, ayah_groups, timing, t,
                                 waveform, wf_rate, font, font_sm)

        # Save as RGB JPEG
        frame_img.convert('RGB').save(out_dir / f'frame_{i+1:05d}.jpg', quality=90)

        if (i + 1) % 200 == 0:
            pct = (i + 1) / len(frame_files) * 100
            print(f"    [{i+1}/{len(frame_files)}] {pct:.0f}%", flush=True)

    print(f"    Rendered {min(len(frame_files), total_frames)} frames", flush=True)

    # 4. Assemble with ffmpeg
    print(f"\n[4] Assembling final video...", flush=True)
    subprocess.run([
        'ffmpeg', '-y',
        '-framerate', str(FPS),
        '-i', str(out_dir / 'frame_%05d.jpg'),
        '-i', str(AUDIO_PATH),
        '-map', '0:v', '-map', '1:a',
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-c:a', 'aac', '-b:a', '128k',
        '-pix_fmt', 'yuv420p',
        '-shortest',
        str(OUTPUT_PATH)
    ], capture_output=True, check=True)

    sz = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\n[5] ✓ Done! {sz:.1f}MB", flush=True)
    print(f"    Output: {OUTPUT_PATH}", flush=True)

    # Cleanup
    shutil.rmtree(FRAMES_DIR)
    print("    Cleaned up temp frames", flush=True)


if __name__ == "__main__":
    main()
