#!/usr/bin/env python3
"""
TajweedSST - Duration Model

Calibrates and validates letter durations based on Tajweed rules.
Works with harakat (beat) counts and reciter-specific speech rates.

Key Features:
- Per-reciter harakat calibration
- Madd type detection from Quranic context
- Duration validation against Tajweed expectations
- Speech rate normalization
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from enum import Enum


class MaddType(Enum):
    NONE = "none"
    ASLI = "asli"           # 2 harakat
    WAJIB = "wajib"         # 4-5 harakat
    JAIZ = "jaiz"           # 2-4-6 harakat (flexible)
    LAZIM = "lazim"         # 6 harakat
    LEEN = "leen"           # 2-4-6 harakat (soft)
    ARID = "arid"           # 2-4-6 harakat (for pause)
    BADAL = "badal"         # 2 harakat (substitution)
    SILAH = "silah"         # 2 harakat (connection)


@dataclass
class HarakatCalibration:
    """Per-reciter timing calibration"""
    reciter_name: str
    harakat_base_ms: float = 100.0      # Base beat duration
    speech_rate_wpm: float = 60.0       # Words per minute
    pitch_range_hz: Tuple[float, float] = (80.0, 300.0)
    sample_size: int = 0                # How many samples used for calibration


@dataclass
class DurationExpectation:
    """Expected duration for a Tajweed rule"""
    rule_name: str
    min_harakat: int
    max_harakat: int
    expected_ms_range: Tuple[float, float]
    tolerance: float = 0.25  # 25% tolerance


@dataclass 
class DurationResult:
    """Result of duration validation"""
    is_valid: bool
    actual_ms: float
    expected_ms: float
    harakat_count: float
    deviation_percent: float
    rule_applied: str


class DurationModel:
    """
    Duration model for Tajweed-based timing validation
    """
    
    # Default expectations (will be calibrated per reciter)
    DEFAULT_HARAKAT_MS = 100.0
    
    # Tajweed duration rules (in harakat counts)
    TAJWEED_DURATIONS = {
        MaddType.ASLI: DurationExpectation("Madd Asli", 2, 2, (150, 280), 0.30),
        MaddType.WAJIB: DurationExpectation("Madd Wajib", 4, 5, (350, 550), 0.25),
        MaddType.LAZIM: DurationExpectation("Madd Lazim", 6, 6, (500, 800), 0.20),
        MaddType.JAIZ: DurationExpectation("Madd Jaiz", 2, 6, (150, 700), 0.30),
        MaddType.ARID: DurationExpectation("Madd Arid", 2, 6, (150, 700), 0.30),
        MaddType.LEEN: DurationExpectation("Madd Leen", 2, 6, (150, 700), 0.30),
    }
    
    # Ghunnah duration
    GHUNNAH_DURATION = DurationExpectation("Ghunnah", 2, 2, (80, 250), 0.30)
    
    def __init__(self, lisan_path: Optional[str] = None):
        """Initialize with optional path to lisan_phonemes.json"""
        self.calibration: Optional[HarakatCalibration] = None
        self.lisan_data: Dict = {}
        
        if lisan_path and Path(lisan_path).exists():
            with open(lisan_path, 'r', encoding='utf-8') as f:
                self.lisan_data = json.load(f)
    
    def calibrate_from_samples(self, 
                                reciter_name: str,
                                vowel_durations: List[float],
                                words_per_minute: float = 60.0) -> HarakatCalibration:
        """
        Calibrate harakat duration from sample vowel measurements
        
        Args:
            reciter_name: Name of reciter for identification
            vowel_durations: List of short vowel durations in seconds
            words_per_minute: Estimated speech rate
        
        Returns:
            HarakatCalibration object
        """
        if not vowel_durations:
            # Use defaults
            self.calibration = HarakatCalibration(
                reciter_name=reciter_name,
                harakat_base_ms=self.DEFAULT_HARAKAT_MS,
                speech_rate_wpm=words_per_minute,
                sample_size=0
            )
            return self.calibration
        
        # Convert to milliseconds and compute median (robust to outliers)
        durations_ms = [d * 1000 for d in vowel_durations]
        harakat_base = np.median(durations_ms)
        
        self.calibration = HarakatCalibration(
            reciter_name=reciter_name,
            harakat_base_ms=harakat_base,
            speech_rate_wpm=words_per_minute,
            sample_size=len(vowel_durations)
        )
        
        return self.calibration
    
    def get_expected_duration(self, 
                               madd_type: MaddType,
                               harakat_count: Optional[int] = None) -> Tuple[float, float]:
        """
        Get expected duration range for a Madd type
        
        Returns:
            Tuple of (min_ms, max_ms)
        """
        if not self.calibration:
            base_ms = self.DEFAULT_HARAKAT_MS
        else:
            base_ms = self.calibration.harakat_base_ms
        
        if madd_type in self.TAJWEED_DURATIONS:
            expectation = self.TAJWEED_DURATIONS[madd_type]
            if harakat_count:
                # Use specific harakat count
                center = harakat_count * base_ms
                tolerance = expectation.tolerance
                return (center * (1 - tolerance), center * (1 + tolerance))
            else:
                # Use range from Tajweed rule
                min_ms = expectation.min_harakat * base_ms * (1 - expectation.tolerance)
                max_ms = expectation.max_harakat * base_ms * (1 + expectation.tolerance)
                return (min_ms, max_ms)
        
        # Default: 1 harakat
        return (base_ms * 0.7, base_ms * 1.3)
    
    def validate_duration(self,
                          actual_duration_s: float,
                          madd_type: MaddType,
                          expected_harakat: int = 2) -> DurationResult:
        """
        Validate if actual duration matches Tajweed expectation
        
        Args:
            actual_duration_s: Actual duration in seconds
            madd_type: Type of Madd rule
            expected_harakat: Expected harakat count
        
        Returns:
            DurationResult with validation details
        """
        actual_ms = actual_duration_s * 1000
        min_ms, max_ms = self.get_expected_duration(madd_type, expected_harakat)
        expected_ms = (min_ms + max_ms) / 2
        
        is_valid = min_ms <= actual_ms <= max_ms
        deviation = abs(actual_ms - expected_ms) / expected_ms * 100 if expected_ms > 0 else 0
        
        # Calculate actual harakat count
        base_ms = self.calibration.harakat_base_ms if self.calibration else self.DEFAULT_HARAKAT_MS
        harakat_count = actual_ms / base_ms if base_ms > 0 else 0
        
        return DurationResult(
            is_valid=is_valid,
            actual_ms=actual_ms,
            expected_ms=expected_ms,
            harakat_count=harakat_count,
            deviation_percent=deviation,
            rule_applied=madd_type.value
        )
    
    def validate_ghunnah_duration(self, actual_duration_s: float) -> DurationResult:
        """Validate Ghunnah duration (2 harakat)"""
        return self.validate_duration(actual_duration_s, MaddType.ASLI, 2)
    
    def suggest_correction(self, 
                            actual_duration_s: float,
                            madd_type: MaddType,
                            expected_harakat: int = 2) -> Tuple[float, float]:
        """
        Suggest corrected start/end times based on Tajweed expectations
        
        Returns:
            Tuple of (suggested_duration_s, adjustment_s)
        """
        min_ms, max_ms = self.get_expected_duration(madd_type, expected_harakat)
        actual_ms = actual_duration_s * 1000
        
        if actual_ms < min_ms:
            # Too short - suggest minimum
            suggested_ms = min_ms
        elif actual_ms > max_ms:
            # Too long - suggest maximum
            suggested_ms = max_ms
        else:
            # Already valid
            suggested_ms = actual_ms
        
        adjustment_ms = suggested_ms - actual_ms
        return (suggested_ms / 1000, adjustment_ms / 1000)
    
    def detect_madd_type_from_context(self,
                                       current_letter: str,
                                       next_letter: Optional[str],
                                       next_harakat: Optional[str],
                                       is_word_end: bool,
                                       is_waqf: bool = False) -> MaddType:
        """
        Auto-detect Madd type from Quranic text context
        
        Args:
            current_letter: The Madd letter (ا و ي)
            next_letter: Following letter (if any)
            next_harakat: Harakat on next letter
            is_word_end: Whether this is at word boundary
            is_waqf: Whether reciter is pausing here
        
        Returns:
            Detected MaddType
        """
        SUKUN = '\u0652'
        SHADDA = '\u0651'
        
        # If at end with pause
        if is_waqf and is_word_end:
            return MaddType.ARID  # Flexible 2-4-6
        
        # Check for Madd Lazim (Sukun or Shadda follows)
        if next_harakat:
            if SHADDA in next_harakat or SUKUN in next_harakat:
                return MaddType.LAZIM
        
        # Check for Madd Wajib (Hamza in same word follows)
        if next_letter and next_letter in 'ءأإؤئ':
            return MaddType.WAJIB
        
        # Default: Madd Asli (natural 2 harakat)
        return MaddType.ASLI


def main():
    """Test duration model"""
    print("=" * 50)
    print("TajweedSST Duration Model Test")
    print("=" * 50)
    
    model = DurationModel()
    
    # Calibrate with sample data (simulated short vowels ~100ms each)
    sample_vowels = [0.095, 0.105, 0.098, 0.102, 0.100, 0.103, 0.097]
    calibration = model.calibrate_from_samples("Abdul_Basit", sample_vowels)
    
    print(f"\nCalibration for {calibration.reciter_name}:")
    print(f"  Harakat base: {calibration.harakat_base_ms:.1f} ms")
    print(f"  Sample size: {calibration.sample_size}")
    
    # Test duration validation
    print("\nDuration Validation Tests:")
    
    # Madd Asli (2 harakat ~ 200ms)
    result = model.validate_duration(0.195, MaddType.ASLI, 2)
    print(f"\n  Madd Asli (0.195s):")
    print(f"    Valid: {result.is_valid}")
    print(f"    Harakat: {result.harakat_count:.1f}")
    print(f"    Deviation: {result.deviation_percent:.1f}%")
    
    # Madd Lazim (6 harakat ~ 600ms)
    result = model.validate_duration(0.580, MaddType.LAZIM, 6)
    print(f"\n  Madd Lazim (0.580s):")
    print(f"    Valid: {result.is_valid}")
    print(f"    Harakat: {result.harakat_count:.1f}")
    print(f"    Deviation: {result.deviation_percent:.1f}%")
    
    # Test Madd type detection
    print("\nMadd Type Detection:")
    detected = model.detect_madd_type_from_context('ا', 'ء', None, False, False)
    print(f"  ا before ء: {detected.value}")
    
    detected = model.detect_madd_type_from_context('ا', 'ب', '\u0651', False, False)
    print(f"  ا before بّ: {detected.value}")


if __name__ == "__main__":
    main()
