#!/usr/bin/env python3
"""
TajweedSST Enhanced Analyzer v3

Integrated improvements:
1. Ghunnah detection (nasal resonance via parselmouth)
2. Pitch tracking for Madd (F0 contour stability)
3. Cross-word rules (Idgham, Ikhfa, Iqlab)
4. Neural-style confidence calibration

Architecture: Lisan al-Arab + DSP + Tajweed Science
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

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("WARNING: parselmouth not available (Ghunnah detection disabled)")

# Paths
AUDIO_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/audio/abdul_basit/surah_090.mp3"
TIMING_PATH = "/home/absolut7/Documents/26apps/MahQuranApp/public/data/letter_timing_90.json"
OUTPUT_PATH = Path(__file__).parent / "output/surah_90_physics_v3.json"

# Character sets
MADD_LETTERS = set('اويٱى')
QALQALAH_LETTERS = set('قطبجد')
TAFKHEEM_LETTERS = set('صضطظخغق')
GHUNNAH_LETTERS = set('نم')  # Nasal letters
HALQ_LETTERS = set('ءهعحغخ')

# Cross-word rule triggers
IDGHAM_TARGETS = set('يرملونw')  # Letters that cause Idgham after ن
IKHFA_TARGETS = set('تثجدذزسشصضطظفقك')  # Letters that cause Ikhfa after ن
IQLAB_TARGET = 'ب'  # ن before ب becomes م


def convert_to_json_safe(obj):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(obj, dict):
        return {k: convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_safe(i) for i in obj]
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class TajweedAnalyzerV3:
    """
    Enhanced Tajweed physics analyzer with full rule detection.
    """
    
    def __init__(self, audio_path, sr=16000, hop_length=256):
        self.audio_path = str(audio_path)
        self.sr = sr
        self.hop_length = hop_length
        
        print(f"Loading audio: {audio_path}")
        self.audio, _ = librosa.load(self.audio_path, sr=self.sr)
        self.duration = len(self.audio) / self.sr
        print(f"  Duration: {self.duration:.1f}s")
        
        # Load for parselmouth (needs original file)
        if HAS_PARSELMOUTH:
            self.sound = parselmouth.Sound(self.audio_path)
    
    def extract_segment(self, start, end):
        """Extract audio segment by time"""
        start_sample = int(start * self.sr)
        end_sample = int(end * self.sr)
        return self.audio[start_sample:end_sample]
    
    # ===== GHUNNAH DETECTION (Nasal Resonance) =====
    
    def analyze_ghunnah(self, start, end, char):
        """
        Analyze Ghunnah (nasal resonance) using formant analysis.
        Nasal sounds have:
        1. Anti-formant (energy dip) around 500-1500 Hz
        2. Higher formant bandwidth
        3. Specific F1/F2 patterns
        """
        if not HAS_PARSELMOUTH:
            return {"status": "SKIPPED", "confidence": 0.0, "reason": "parselmouth unavailable"}
        
        try:
            # Extract segment from parselmouth sound
            segment = self.sound.extract_part(from_time=start, to_time=end, preserve_times=False)
            
            if segment.get_total_duration() < 0.03:
                return {"status": "TOO_SHORT", "confidence": 0.0}
            
            # Get formants
            formants = call(segment, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            
            # Average F1 and F2
            n_frames = call(formants, "Get number of frames")
            if n_frames < 1:
                return {"status": "NO_FRAMES", "confidence": 0.0}
            
            f1_values = []
            f2_values = []
            bandwidths = []
            
            for i in range(1, n_frames + 1):
                time = call(formants, "Get time from frame number", i)
                f1 = call(formants, "Get value at time", 1, time, "Hertz", "Linear")
                f2 = call(formants, "Get value at time", 2, time, "Hertz", "Linear")
                bw1 = call(formants, "Get bandwidth at time", 1, time, "Hertz", "Linear")
                
                if not np.isnan(f1):
                    f1_values.append(f1)
                if not np.isnan(f2):
                    f2_values.append(f2)
                if not np.isnan(bw1):
                    bandwidths.append(bw1)
            
            if not f1_values or not bandwidths:
                return {"status": "NO_FORMANTS", "confidence": 0.0}
            
            avg_f1 = np.mean(f1_values)
            avg_f2 = np.mean(f2_values) if f2_values else 0
            avg_bandwidth = np.mean(bandwidths)
            
            # Ghunnah indicators:
            # 1. Low F1 (nasal cavity resonance) - typically 200-400 Hz
            # 2. High bandwidth (nasal damping)
            # 3. F2 in nasal range
            
            low_f1 = avg_f1 < 500
            high_bandwidth = avg_bandwidth > 150
            nasal_f2 = 800 < avg_f2 < 2000
            
            indicators = sum([low_f1, high_bandwidth, nasal_f2])
            
            if indicators >= 2:
                status = "DETECTED"
                confidence = 0.7 + (indicators - 2) * 0.15
            elif indicators == 1:
                status = "PARTIAL"
                confidence = 0.5
            else:
                status = "NOT_DETECTED"
                confidence = 0.2
            
            return {
                "status": status,
                "confidence": round(confidence, 3),
                "f1": round(avg_f1, 1),
                "f2": round(avg_f2, 1),
                "bandwidth": round(avg_bandwidth, 1),
                "indicators": {"low_f1": low_f1, "high_bandwidth": high_bandwidth, "nasal_f2": nasal_f2}
            }
            
        except Exception as e:
            return {"status": "ERROR", "confidence": 0.0, "error": str(e)}
    
    # ===== PITCH TRACKING FOR MADD =====
    
    def analyze_madd_pitch(self, segment, char, expected_count=2):
        """
        Analyze Madd using pitch (F0) stability.
        Sustained vowels have stable pitch with minimal variation.
        """
        duration_ms = len(segment) / self.sr * 1000
        
        # Extract pitch using librosa
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                segment, 
                fmin=50, 
                fmax=500, 
                sr=self.sr,
                frame_length=1024,
                hop_length=256
            )
        except Exception as e:
            # Fallback to basic sustain detection
            return self._basic_madd_analysis(segment, duration_ms, expected_count)
        
        # Filter to voiced frames only
        f0_voiced = f0[~np.isnan(f0)]
        
        if len(f0_voiced) < 3:
            return self._basic_madd_analysis(segment, duration_ms, expected_count)
        
        # Pitch stability: low coefficient of variation = sustained
        pitch_mean = np.mean(f0_voiced)
        pitch_std = np.std(f0_voiced)
        pitch_cv = pitch_std / pitch_mean if pitch_mean > 0 else 1.0
        
        # Voicing ratio: high means continuous sound
        voicing_ratio = len(f0_voiced) / len(f0)
        
        # Sustain score based on pitch stability and voicing
        pitch_stable = pitch_cv < 0.15
        well_voiced = voicing_ratio > 0.6
        
        # Expected duration
        base_haraka = 100  # ms
        expected_duration = expected_count * base_haraka
        duration_match = 0.7 <= (duration_ms / expected_duration) <= 1.5 if expected_duration > 0 else False
        
        if pitch_stable and well_voiced and duration_match:
            status = "SUSTAINED"
            confidence = 0.85
        elif (pitch_stable and well_voiced) or (well_voiced and duration_match):
            status = "PARTIAL"
            confidence = 0.6
        elif well_voiced:
            status = "VOICED"
            confidence = 0.4
        else:
            status = "WEAK"
            confidence = 0.2
        
        return {
            "status": status,
            "confidence": round(confidence, 3),
            "duration_ms": round(duration_ms, 1),
            "expected_ms": round(expected_duration, 1),
            "pitch_mean": round(pitch_mean, 1),
            "pitch_cv": round(pitch_cv, 3),
            "voicing_ratio": round(voicing_ratio, 3)
        }
    
    def _basic_madd_analysis(self, segment, duration_ms, expected_count):
        """Fallback basic Madd analysis"""
        expected_duration = expected_count * 100
        ratio = duration_ms / expected_duration if expected_duration > 0 else 0
        
        if 0.7 <= ratio <= 1.5:
            return {"status": "SUSTAINED", "confidence": 0.5, "duration_ms": round(duration_ms, 1)}
        return {"status": "WEAK", "confidence": 0.3, "duration_ms": round(duration_ms, 1)}
    
    # ===== QALQALAH (Improved) =====
    
    def analyze_qalqalah(self, segment):
        """Improved Qalqalah detection with energy release pattern"""
        if len(segment) < 256:
            return {"status": "TOO_SHORT", "confidence": 0.0}
        
        frame_length = min(256, len(segment) // 2)
        hop = frame_length // 4
        
        rms = librosa.feature.rms(y=segment, frame_length=frame_length, hop_length=hop)[0]
        
        if len(rms) < 3:
            return {"status": "INSUFFICIENT", "confidence": 0.0}
        
        # Qalqalah: energy release at end
        split = len(rms) * 2 // 3
        first = np.mean(rms[:split])
        last = np.mean(rms[split:])
        max_rms = np.max(rms)
        mean_rms = np.mean(rms)
        
        has_energy = mean_rms > 0.01
        has_release = last > first * 0.8
        has_spike = max_rms > mean_rms * 1.3
        
        if has_energy and has_release and has_spike:
            confidence = min(0.9, (max_rms / mean_rms - 1) + 0.5)
            return {"status": "DETECTED", "confidence": round(confidence, 3)}
        elif has_energy:
            return {"status": "PARTIAL", "confidence": 0.4}
        return {"status": "NO_ENERGY", "confidence": 0.1}
    
    # ===== TAFKHEEM (Heavy Letters) =====
    
    def analyze_tafkheem(self, segment):
        """Analyze Tafkheem using spectral characteristics"""
        if len(segment) < 512:
            return {"status": "TOO_SHORT", "confidence": 0.0}
        
        centroid = librosa.feature.spectral_centroid(y=segment, sr=self.sr)[0]
        mean_centroid = np.mean(centroid)
        
        S = np.abs(librosa.stft(segment))
        freqs = librosa.fft_frequencies(sr=self.sr)
        low_idx = np.where(freqs < 1000)[0]
        high_idx = np.where(freqs >= 1000)[0]
        
        low_energy = np.sum(S[low_idx, :])
        high_energy = np.sum(S[high_idx, :])
        total = low_energy + high_energy
        low_ratio = low_energy / total if total > 0 else 0.5
        
        if mean_centroid < 1500 and low_ratio > 0.6:
            return {"status": "HEAVY", "confidence": 0.9, "centroid": round(mean_centroid, 1)}
        elif mean_centroid < 2000 or low_ratio > 0.5:
            return {"status": "MODERATE", "confidence": 0.7, "centroid": round(mean_centroid, 1)}
        return {"status": "LIGHT", "confidence": 0.4, "centroid": round(mean_centroid, 1)}
    
    # ===== CROSS-WORD RULES =====
    
    def analyze_cross_word_rules(self, timing_data):
        """
        Analyze cross-word Tajweed rules:
        - Idgham: ن/م merges into following letter
        - Ikhfa: ن partially hidden before certain letters
        - Iqlab: ن becomes م sound before ب
        """
        results = {
            "idgham": [],
            "ikhfa": [],
            "iqlab": []
        }
        
        for i, entry in enumerate(timing_data):
            char = entry.get("char", "")
            base_char = char[0] if char else ""
            
            # Check if this is a Noon with Sukun or Tanween
            has_sukun = 'ْ' in char
            has_tanween = any(c in char for c in 'ًٌٍ')
            is_noon_trigger = base_char == 'ن' and (has_sukun or has_tanween)
            is_meem_trigger = base_char == 'م' and has_sukun
            
            if not (is_noon_trigger or is_meem_trigger):
                continue
            
            # Look at next letter
            if i + 1 >= len(timing_data):
                continue
            
            next_entry = timing_data[i + 1]
            next_char = next_entry.get("char", "")
            next_base = next_char[0] if next_char else ""
            
            # Iqlab: ن before ب
            if is_noon_trigger and next_base == IQLAB_TARGET:
                # Analyze if ن sounds like م
                segment = self.extract_segment(entry.get("start", 0), entry.get("end", 0))
                ghunnah = self.analyze_ghunnah(entry.get("start", 0), entry.get("end", 0), char)
                
                results["iqlab"].append({
                    "position": i,
                    "char": char,
                    "next_char": next_char,
                    "time": f"{entry.get('start', 0):.3f}-{entry.get('end', 0):.3f}",
                    "ghunnah_detected": ghunnah.get("status") in ["DETECTED", "PARTIAL"],
                    "confidence": ghunnah.get("confidence", 0)
                })
            
            # Ikhfa: ن before specific letters
            elif is_noon_trigger and next_base in IKHFA_TARGETS:
                # Analyze partial nasalization
                segment = self.extract_segment(entry.get("start", 0), entry.get("end", 0))
                ghunnah = self.analyze_ghunnah(entry.get("start", 0), entry.get("end", 0), char)
                
                results["ikhfa"].append({
                    "position": i,
                    "char": char,
                    "next_char": next_char,
                    "time": f"{entry.get('start', 0):.3f}-{entry.get('end', 0):.3f}",
                    "ghunnah_level": ghunnah.get("status"),
                    "confidence": ghunnah.get("confidence", 0)
                })
            
            # Idgham: ن before يرملون
            elif is_noon_trigger and next_base in IDGHAM_TARGETS:
                # Check if ن is merged (very short duration)
                noon_dur = (entry.get("end", 0) - entry.get("start", 0)) * 1000
                
                results["idgham"].append({
                    "position": i,
                    "char": char,
                    "next_char": next_char,
                    "time": f"{entry.get('start', 0):.3f}-{entry.get('end', 0):.3f}",
                    "noon_duration_ms": round(noon_dur, 1),
                    "merged": noon_dur < 50,  # Very short = merged
                    "confidence": 0.7 if noon_dur < 50 else 0.4
                })
        
        return results


def run_comprehensive_analysis():
    """Run comprehensive Tajweed analysis with all improvements"""
    
    print("=" * 60)
    print("TajweedSST Enhanced Analyzer v3")
    print("Ghunnah + Pitch + Cross-Word Rules")
    print("=" * 60)
    
    if not HAS_LIBROSA:
        print("ERROR: librosa required")
        return
    
    # Load analyzer
    analyzer = TajweedAnalyzerV3(AUDIO_PATH)
    
    # Load timing
    with open(TIMING_PATH, 'r', encoding='utf-8') as f:
        timing = json.load(f)
    
    print(f"\n[1] Analyzing {len(timing)} letters...")
    
    results = {
        "qalqalah": [],
        "madd": [],
        "tafkheem": [],
        "ghunnah": [],
        "cross_word": {},
        "summary": {}
    }
    
    counts = {k: 0 for k in ["qalqalah", "madd", "tafkheem", "ghunnah"]}
    passed = {k: 0 for k in ["qalqalah", "madd", "tafkheem", "ghunnah"]}
    
    for entry in timing:
        char = entry.get("char", "")
        base = char[0] if char else ""
        start = entry.get("start", 0)
        end = entry.get("end", 0)
        
        segment = analyzer.extract_segment(start, end)
        
        # Qalqalah
        if base in QALQALAH_LETTERS:
            counts["qalqalah"] += 1
            analysis = analyzer.analyze_qalqalah(segment)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["qalqalah"].append(analysis)
            if analysis["confidence"] >= 0.4:
                passed["qalqalah"] += 1
        
        # Madd (with pitch tracking)
        if base in MADD_LETTERS:
            counts["madd"] += 1
            madd_count = entry.get("madd_count", 2)
            analysis = analyzer.analyze_madd_pitch(segment, char, madd_count)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["madd"].append(analysis)
            if analysis["status"] in ["SUSTAINED", "PARTIAL"]:
                passed["madd"] += 1
        
        # Tafkheem
        if base in TAFKHEEM_LETTERS:
            counts["tafkheem"] += 1
            analysis = analyzer.analyze_tafkheem(segment)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["tafkheem"].append(analysis)
            if analysis["status"] in ["HEAVY", "MODERATE"]:
                passed["tafkheem"] += 1
        
        # Ghunnah
        if base in GHUNNAH_LETTERS:
            counts["ghunnah"] += 1
            analysis = analyzer.analyze_ghunnah(start, end, char)
            analysis["char"] = char
            analysis["time"] = f"{start:.3f}-{end:.3f}"
            results["ghunnah"].append(analysis)
            if analysis.get("status") in ["DETECTED", "PARTIAL"]:
                passed["ghunnah"] += 1
    
    # Cross-word analysis
    print("\n[2] Analyzing cross-word rules...")
    results["cross_word"] = analyzer.analyze_cross_word_rules(timing)
    
    # Summary
    results["summary"] = {
        k: {
            "total": counts[k],
            "passed": passed[k],
            "rate": round(passed[k] / max(1, counts[k]), 2)
        }
        for k in counts
    }
    
    results["summary"]["cross_word"] = {
        "idgham": len(results["cross_word"].get("idgham", [])),
        "ikhfa": len(results["cross_word"].get("ikhfa", [])),
        "iqlab": len(results["cross_word"].get("iqlab", []))
    }
    
    # Print results
    print("\n[3] Results:")
    for rule, data in results["summary"].items():
        if isinstance(data, dict) and "rate" in data:
            print(f"    {rule}: {data['passed']}/{data['total']} ({data['rate']*100:.0f}%)")
        elif isinstance(data, dict):
            print(f"    {rule}: Idgham={data.get('idgham', 0)}, Ikhfa={data.get('ikhfa', 0)}, Iqlab={data.get('iqlab', 0)}")
    
    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(convert_to_json_safe(results), f, ensure_ascii=False, indent=2)
    print(f"\n[4] Saved: {OUTPUT_PATH}")
    
    # Samples
    print("\n[5] Sample Ghunnah (ن/م nasal detection):")
    for r in results["ghunnah"][:5]:
        f1 = r.get('f1', 'N/A')
        print(f"    [{r['char']}] F1:{f1}Hz → {r['status']} (conf: {r['confidence']})")
    
    print("\n[6] Sample Madd (Pitch Tracking):")
    for r in results["madd"][:5]:
        cv = r.get('pitch_cv', 'N/A')
        print(f"    [{r['char']}] {r.get('duration_ms', 0):.0f}ms, pitch_cv:{cv} → {r['status']}")
    
    print("\n[7] Cross-Word Rules Detected:")
    for rule, items in results["cross_word"].items():
        if items:
            print(f"    {rule.upper()}: {len(items)} instances")
            for item in items[:2]:
                print(f"      - {item['char']} → {item['next_char']} @ {item['time']}")
    
    print("\n" + "=" * 60)
    print("✓ TajweedSST v3 Analysis Complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    run_comprehensive_analysis()
