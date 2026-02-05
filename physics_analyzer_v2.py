#!/usr/bin/env python3
"""
Enhanced Physics Wave Analyzer - Using Lisan al-Arab Principles

Integrated from MahQuranApp/scripts/lisan_madd_detector.py

Key techniques:
1. Sustained region detection (spectral flux + energy stability)
2. Anti-drift stabilization (gap closing + minimum duration)
3. Per-character Tajweed physics analysis
"""

import json
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("WARNING: librosa not available")

# Paths
AUDIO_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/audio/abdul_basit/surah_090.mp3"
TIMING_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/data/letter_timing_90.json"
OUTPUT_PATH = Path(__file__).parent / "output/surah_90_physics_v2.json"

# Tajweed character sets
MADD_LETTERS = set('اويٱى')
QALQALAH_LETTERS = set('قطبجد')
TAFKHEEM_LETTERS = set('صضطظخغق')
HALQ_LETTERS = set('ءهعحغخ')


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


class LisanPhysicsAnalyzer:
    """
    Physics analyzer using Lisan al-Arab acoustic principles.
    """
    
    def __init__(self, audio_path, sr=16000, hop_length=256):
        self.audio_path = str(audio_path)
        self.sr = sr
        self.hop_length = hop_length
        
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        self.duration = len(self.audio) / self.sr
        print(f"  Duration: {self.duration:.1f}s, Sample rate: {sr}Hz")
    
    def extract_segment(self, start, end):
        """Extract audio segment by time"""
        start_sample = int(start * self.sr)
        end_sample = int(end * self.sr)
        return self.audio[start_sample:end_sample]
    
    def detect_sustained_regions(self, segment):
        """
        Detect regions where sound is SUSTAINED (استمرّ).
        From LisanMaddDetector - detects madd vowels being held.
        
        Returns: array of sustain scores per frame (higher = more sustained)
        """
        if len(segment) < 512:
            return np.zeros(1)
        
        # 1. Compute spectral flux (low flux = sustained sound)
        S = np.abs(librosa.stft(segment, hop_length=self.hop_length))
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))
        flux = np.concatenate([[0], flux])
        flux = gaussian_filter1d(flux.astype(np.float64), sigma=2)
        
        # Invert: high score where flux is LOW (sustained sound)
        max_flux = np.max(flux) if np.max(flux) > 0 else 1
        sustain_score = 1 - (flux / max_flux)
        
        # 2. Check energy stability (sustained sounds have stable RMS)
        energy = librosa.feature.rms(y=segment, hop_length=self.hop_length)[0]
        energy = gaussian_filter1d(energy.astype(np.float64), sigma=2)
        
        # Energy stability: low variance in local windows
        stability = np.zeros_like(energy)
        window = 5
        for i in range(window, len(energy) - window):
            local_std = np.std(energy[max(0, i-window):i+window])
            local_mean = np.mean(energy[max(0, i-window):i+window])
            if local_mean > 0:
                stability[i] = 1 - min(local_std / local_mean, 1)
        
        # Pad stability to match sustain_score length
        min_len = min(len(sustain_score), len(stability))
        sustain_score = sustain_score[:min_len]
        stability = stability[:min_len]
        
        # Combined score: both low flux AND stable energy = sustained vowel
        combined = sustain_score * stability
        
        return combined
    
    def analyze_madd(self, segment, char, expected_count=2):
        """
        Analyze Madd (elongation) using sustain detection.
        """
        duration_ms = len(segment) / self.sr * 1000
        
        # Detect sustained regions
        sustain_scores = self.detect_sustained_regions(segment)
        avg_sustain = np.mean(sustain_scores) if len(sustain_scores) > 0 else 0
        
        # Calculate expected duration
        base_haraka = 100  # ms per haraka (Abdul Basit is slower)
        expected_duration = expected_count * base_haraka
        
        # Determine if sustain matches expected madd
        if avg_sustain > 0.5:
            detected_count = 3 if avg_sustain > 0.7 else 2
        else:
            detected_count = 1
        
        ratio = duration_ms / expected_duration if expected_duration > 0 else 0
        
        if ratio >= 0.7 and avg_sustain >= 0.4:
            status = "SUSTAINED"
            confidence = 0.8 if avg_sustain > 0.6 else 0.6
        elif ratio >= 0.5:
            status = "PARTIAL"
            confidence = 0.5
        else:
            status = "SHORT"
            confidence = 0.3
        
        return {
            "status": status,
            "confidence": round(confidence, 3),
            "actual_ms": round(duration_ms, 1),
            "expected_ms": round(expected_duration, 1),
            "ratio": round(ratio, 2),
            "sustain_score": round(avg_sustain, 3),
            "detected_count": detected_count
        }
    
    def analyze_qalqalah(self, segment):
        """
        Analyze Qalqalah (bounce) using RMS energy patterns.
        Improved: checks for energy release at end of segment.
        """
        if len(segment) < 256:
            return {"status": "TOO_SHORT", "confidence": 0.0}
        
        # Use smaller frame for short segments
        frame_length = min(256, len(segment) // 2)
        hop = frame_length // 4
        
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop)[0]
        
        if len(rms) < 3:
            return {"status": "INSUFFICIENT_FRAMES", "confidence": 0.0}
        
        # Qalqalah pattern: should have energy release at end
        # Look at last third vs first two-thirds
        split_idx = len(rms) * 2 // 3
        first_part = np.mean(rms[:split_idx])
        last_part = np.mean(rms[split_idx:])
        
        # Also check for any spike in segment
        max_rms = np.max(rms)
        mean_rms = np.mean(rms)
        
        has_energy = mean_rms > 0.01
        has_release = last_part > first_part * 0.8  # Energy maintained or released at end
        has_spike = max_rms > mean_rms * 1.3
        
        if has_energy and has_release and has_spike:
            confidence = min(0.9, (max_rms / mean_rms - 1) + 0.5)
            return {
                "status": "DETECTED",
                "confidence": round(confidence, 3),
                "pattern": {
                    "first": round(float(first_part), 4),
                    "last": round(float(last_part), 4),
                    "max": round(float(max_rms), 4),
                    "mean": round(float(mean_rms), 4)
                }
            }
        elif has_energy:
            return {"status": "PARTIAL", "confidence": 0.4}
        else:
            return {"status": "NO_ENERGY", "confidence": 0.1}
    
    def analyze_tafkheem(self, segment):
        """
        Analyze Tafkheem (heaviness) using spectral centroid.
        Heavy consonants have lower spectral centroid (more bass).
        """
        if len(segment) < 512:
            return {"status": "TOO_SHORT", "confidence": 0.0}
        
        # Compute spectral centroid
        centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0]
        mean_centroid = np.mean(centroid)
        
        # Also check low-frequency energy ratio
        S = np.abs(librosa.stft(segment))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_freq_idx = np.where(freqs < 1000)[0]
        high_freq_idx = np.where(freqs >= 1000)[0]
        
        low_energy = np.sum(S[low_freq_idx, :])
        high_energy = np.sum(S[high_freq_idx, :])
        total_energy = low_energy + high_energy
        
        low_ratio = low_energy / total_energy if total_energy > 0 else 0.5
        
        # Heavy letters: low centroid + high low-frequency ratio
        if mean_centroid < 1500 and low_ratio > 0.6:
            status = "HEAVY"
            confidence = 0.9
        elif mean_centroid < 2000 or low_ratio > 0.5:
            status = "MODERATE"
            confidence = 0.7
        else:
            status = "LIGHT"
            confidence = 0.4
        
        return {
            "status": status,
            "confidence": round(confidence, 3),
            "spectral_centroid": round(float(mean_centroid), 1),
            "low_freq_ratio": round(float(low_ratio), 3)
        }


def run_enhanced_analysis():
    """Run enhanced physics analysis on all tagged letters"""
    
    print("=" * 60)
    print("Enhanced Physics Analysis - Surah 90")
    print("Using Lisan al-Arab Acoustic Principles")
    print("=" * 60)
    
    if not HAS_LIBROSA:
        print("ERROR: librosa required for analysis")
        return
    
    # Load analyzer
    analyzer = LisanPhysicsAnalyzer(AUDIO_PATH)
    
    # Load timing data
    with open(TIMING_PATH, 'r', encoding='utf-8') as f:
        timing = json.load(f)
    
    print(f"\n[1] Analyzing {len(timing)} letters...")
    
    # Results
    results = {
        "qalqalah": [],
        "madd": [],
        "tafkheem": [],
        "summary": {}
    }
    
    counts = {"qalqalah": 0, "madd": 0, "tafkheem": 0}
    passed = {"qalqalah": 0, "madd": 0, "tafkheem": 0}
    
    for entry in timing:
        char = entry.get("char", "")
        base_char = char[0] if char else ""  # First char is base letter
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        
        segment = analyzer.extract_segment(start, end)
        
        # Analyze based on character type
        if base_char in QALQALAH_LETTERS:
            counts["qalqalah"] += 1
            analysis = analyzer.analyze_qalqalah(segment)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["qalqalah"].append(analysis)
            if analysis["confidence"] >= 0.4:
                passed["qalqalah"] += 1
        
        if base_char in MADD_LETTERS:
            counts["madd"] += 1
            madd_count = entry.get("madd_count", 2)
            analysis = analyzer.analyze_madd(segment, char, madd_count)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["madd"].append(analysis)
            if analysis["status"] in ["SUSTAINED", "PARTIAL"]:
                passed["madd"] += 1
        
        if base_char in TAFKHEEM_LETTERS:
            counts["tafkheem"] += 1
            analysis = analyzer.analyze_tafkheem(segment)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["tafkheem"].append(analysis)
            if analysis["status"] in ["HEAVY", "MODERATE"]:
                passed["tafkheem"] += 1
    
    # Summary
    results["summary"] = {
        "qalqalah": {
            "total": counts["qalqalah"],
            "passed": passed["qalqalah"],
            "rate": round(passed["qalqalah"] / max(1, counts["qalqalah"]), 2)
        },
        "madd": {
            "total": counts["madd"],
            "passed": passed["madd"],
            "rate": round(passed["madd"] / max(1, counts["madd"]), 2)
        },
        "tafkheem": {
            "total": counts["tafkheem"],
            "passed": passed["tafkheem"],
            "rate": round(passed["tafkheem"] / max(1, counts["tafkheem"]), 2)
        },
    }
    
    # Print results
    print("\n[2] Results (Using Lisan Acoustic Detection):")
    print(f"    Qalqalah: {passed['qalqalah']}/{counts['qalqalah']} ({results['summary']['qalqalah']['rate']*100:.0f}%)")
    print(f"    Madd: {passed['madd']}/{counts['madd']} ({results['summary']['madd']['rate']*100:.0f}%)")
    print(f"    Tafkheem: {passed['tafkheem']}/{counts['tafkheem']} ({results['summary']['tafkheem']['rate']*100:.0f}%)")
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_safe(results), f, ensure_ascii=False, indent=2)
    print(f"\n[3] Saved: {OUTPUT_PATH}")
    
    # Show samples
    print("\n[4] Sample Qalqalah (Improved Detection):")
    for r in results["qalqalah"][:5]:
        print(f"    [{r['char']}] {r['time']} → {r['status']} (conf: {r['confidence']})")
    
    print("\n[5] Sample Madd (Sustain Detection):")
    for r in results["madd"][:5]:
        print(f"    [{r['char']}] {r['actual_ms']:.0f}ms, sustain:{r['sustain_score']:.2f} → {r['status']}")
    
    print("\n[6] Sample Tafkheem (Heavy Letter Detection):")
    for r in results["tafkheem"][:5]:
        print(f"    [{r['char']}] centroid:{r['spectral_centroid']:.0f}Hz, low_ratio:{r['low_freq_ratio']:.2f} → {r['status']}")
    
    print("\n" + "=" * 60)
    print("✓ Enhanced Physics Analysis Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_enhanced_analysis()
