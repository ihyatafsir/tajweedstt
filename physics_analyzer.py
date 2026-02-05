#!/usr/bin/env python3
"""
Physics Wave Analyzer for Surah 90

Validates Tajweed rules using actual audio signal processing:
- Qalqalah: RMS energy dip→spike pattern
- Madd: Duration verification (2x, 4x, 6x average)  
- Tafkheem: Low-frequency energy presence
"""

import json
import numpy as np
from pathlib import Path

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not available")


def convert_to_json_safe(obj):
    """Convert numpy types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {k: convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Paths
AUDIO_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/audio/abdul_basit/surah_090.mp3"
TIMING_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/data/letter_timing_90.json"
OUTPUT_PATH = Path(__file__).parent / "output/surah_90_physics.json"


def load_audio():
    """Load audio file"""
    print(f"Loading: {AUDIO_PATH}")
    y, sr = librosa.load(AUDIO_PATH, sr=22050)
    duration = len(y) / sr
    print(f"  Duration: {duration:.1f}s, Sample rate: {sr}Hz")
    return y, sr


def load_timing():
    """Load timing data with Tajweed tags"""
    with open(TIMING_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_segment(y, sr, start, end):
    """Extract audio segment"""
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return y[start_sample:end_sample]


def analyze_qalqalah(segment, sr):
    """
    Analyze Qalqalah (bounce) pattern.
    Expected: RMS dip followed by spike at letter end.
    """
    if len(segment) < 512:
        return {"status": "TOO_SHORT", "confidence": 0.0}
    
    # Calculate RMS energy
    rms = librosa.feature.rms(y=segment, frame_length=256, hop_length=64)[0]
    
    if len(rms) < 4:
        return {"status": "INSUFFICIENT_FRAMES", "confidence": 0.0}
    
    # Look for dip→spike pattern
    # Divide into thirds
    third = len(rms) // 3
    if third < 1:
        return {"status": "TOO_SHORT", "confidence": 0.0}
    
    first_third = np.mean(rms[:third])
    middle_third = np.mean(rms[third:2*third])
    last_third = np.mean(rms[2*third:])
    
    # Qalqalah pattern: middle should dip, end should spike
    has_dip = middle_third < first_third * 0.9
    has_spike = last_third > middle_third * 1.1
    
    if has_dip and has_spike:
        confidence = min(1.0, (first_third - middle_third) / first_third + (last_third - middle_third) / last_third)
        return {
            "status": "DETECTED",
            "confidence": round(confidence, 3),
            "pattern": {"first": round(float(first_third), 4), "middle": round(float(middle_third), 4), "last": round(float(last_third), 4)}
        }
    elif has_spike:
        return {"status": "PARTIAL_SPIKE", "confidence": 0.5}
    else:
        return {"status": "NOT_DETECTED", "confidence": 0.2}


def analyze_madd(segment, sr, expected_count):
    """
    Analyze Madd (elongation) duration.
    Verify letter duration matches expected count (2, 4, or 6 harakaat).
    """
    duration_ms = len(segment) / sr * 1000
    
    # Average haraka duration ~100-150ms for Tarteel recitation
    base_haraka = 120  # ms
    expected_duration = expected_count * base_haraka
    
    ratio = duration_ms / expected_duration if expected_duration > 0 else 0
    
    # Allow ±30% tolerance
    if 0.7 <= ratio <= 1.3:
        status = "CORRECT"
        confidence = 1.0 - abs(1.0 - ratio)
    elif 0.5 <= ratio <= 1.5:
        status = "CLOSE"
        confidence = 0.6
    else:
        status = "MISMATCH"
        confidence = 0.3
    
    return {
        "status": status,
        "confidence": round(confidence, 3),
        "actual_ms": round(duration_ms, 1),
        "expected_ms": round(expected_duration, 1),
        "ratio": round(ratio, 2)
    }


def analyze_tafkheem(segment, sr):
    """
    Analyze Tafkheem (heaviness) - heavy letters have stronger low frequencies.
    """
    if len(segment) < 1024:
        return {"status": "TOO_SHORT", "confidence": 0.0}
    
    # Compute spectral centroid - lower = heavier
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr)[0]
    mean_centroid = np.mean(centroid)
    
    # Heavy letters typically have centroid < 2000Hz
    # Light letters typically > 2500Hz
    if mean_centroid < 1800:
        status = "HEAVY"
        confidence = 0.9
    elif mean_centroid < 2200:
        status = "MODERATE"
        confidence = 0.7
    else:
        status = "LIGHT"
        confidence = 0.4
    
    return {
        "status": status,
        "confidence": round(confidence, 3),
        "spectral_centroid": round(float(mean_centroid), 1)
    }


def run_analysis():
    """Run physics analysis on all tagged letters"""
    
    print("=" * 60)
    print("Physics Wave Analysis - Surah 90")
    print("=" * 60)
    
    if not HAS_LIBROSA:
        print("ERROR: librosa required for analysis")
        return
    
    # Load data
    y, sr = load_audio()
    timing = load_timing()
    
    print(f"\n[1] Analyzing {len(timing)} letters...")
    
    # Analyze each tagged letter
    results = {
        "qalqalah": [],
        "madd": [],
        "tafkheem": [],
        "summary": {}
    }
    
    counts = {"qalqalah": 0, "madd": 0, "tafkheem": 0, "other": 0}
    passed = {"qalqalah": 0, "madd": 0, "tafkheem": 0}
    
    for entry in timing:
        tajweed = entry.get("tajweed_type", "None")
        physics = entry.get("physics_check", "None")
        
        if tajweed == "None" or physics == "None":
            continue
        
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        char = entry.get("char", "")
        
        segment = extract_segment(y, sr, start, end)
        
        if "qalqalah" in tajweed.lower():
            counts["qalqalah"] += 1
            analysis = analyze_qalqalah(segment, sr)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            analysis["tajweed"] = tajweed
            results["qalqalah"].append(analysis)
            if analysis["confidence"] >= 0.5:
                passed["qalqalah"] += 1
        
        elif "madd" in tajweed.lower():
            counts["madd"] += 1
            madd_count = entry.get("madd_count", 2)
            analysis = analyze_madd(segment, sr, madd_count)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            analysis["tajweed"] = tajweed
            analysis["expected_count"] = madd_count
            results["madd"].append(analysis)
            if analysis["confidence"] >= 0.5:
                passed["madd"] += 1
        
        elif "tafkheem" in tajweed.lower():
            counts["tafkheem"] += 1
            analysis = analyze_tafkheem(segment, sr)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            analysis["tajweed"] = tajweed
            results["tafkheem"].append(analysis)
            if analysis["status"] in ["HEAVY", "MODERATE"]:
                passed["tafkheem"] += 1
        
        else:
            counts["other"] += 1
    
    # Summary
    results["summary"] = {
        "qalqalah": {"total": counts["qalqalah"], "passed": passed["qalqalah"], "rate": round(passed["qalqalah"]/max(1,counts["qalqalah"]), 2)},
        "madd": {"total": counts["madd"], "passed": passed["madd"], "rate": round(passed["madd"]/max(1,counts["madd"]), 2)},
        "tafkheem": {"total": counts["tafkheem"], "passed": passed["tafkheem"], "rate": round(passed["tafkheem"]/max(1,counts["tafkheem"]), 2)},
    }
    
    # Print results
    print("\n[2] Results:")
    print(f"    Qalqalah: {passed['qalqalah']}/{counts['qalqalah']} passed ({results['summary']['qalqalah']['rate']*100:.0f}%)")
    print(f"    Madd: {passed['madd']}/{counts['madd']} passed ({results['summary']['madd']['rate']*100:.0f}%)")
    print(f"    Tafkheem: {passed['tafkheem']}/{counts['tafkheem']} passed ({results['summary']['tafkheem']['rate']*100:.0f}%)")
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_safe(results), f, ensure_ascii=False, indent=2)
    print(f"\n[3] Saved: {OUTPUT_PATH}")
    
    # Show samples
    print("\n[4] Sample Qalqalah Analysis:")
    for r in results["qalqalah"][:3]:
        print(f"    [{r['char']}] {r['time']} → {r['status']} (conf: {r['confidence']})")
    
    print("\n[5] Sample Madd Analysis:")
    for r in results["madd"][:3]:
        print(f"    [{r['char']}] {r['actual_ms']:.0f}ms vs {r['expected_ms']:.0f}ms → {r['status']}")
    
    print("\n" + "=" * 60)
    print("✓ Physics Analysis Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_analysis()
