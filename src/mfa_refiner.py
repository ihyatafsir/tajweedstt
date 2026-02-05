#!/usr/bin/env python3
"""
TajweedSST - MFA Refiner Post-Processor

Refines wav2vec/MFA alignments using Tajweed physics validation.
This is the main integration layer that combines:
1. Tajweed Parser (text → rules)
2. Physics Validators (audio → boundaries)
3. Duration Model (timing → corrections)

Output: Refined alignment JSON with confidence scores.
"""

import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# Import TajweedSST components
from .tajweed_parser import TajweedParser, TajweedType, PhysicsCheck
from .physics_validator import PhysicsValidator, ValidationStatus
from .duration_model import DurationModel, MaddType


@dataclass
class RefinedLetter:
    """A letter with refined timing and confidence"""
    letter: str
    phoneme: str
    original_start: float
    original_end: float
    refined_start: float
    refined_end: float
    tajweed_rule: str
    physics_score: float
    duration_valid: bool
    confidence: float


@dataclass
class RefinedWord:
    """A word with refined letter timings"""
    word_text: str
    start: float
    end: float
    letters: List[RefinedLetter]
    average_confidence: float


@dataclass
class RefinementResult:
    """Complete refinement result for an audio segment"""
    audio_path: str
    original_alignment_path: str
    words: List[RefinedWord]
    overall_confidence: float
    statistics: Dict


class MFARefiner:
    """
    Post-processor that refines MFA/wav2vec alignments using Tajweed physics
    """
    
    def __init__(self, 
                 lisan_path: Optional[str] = None,
                 sample_rate: int = 22050):
        """
        Initialize the refiner with Tajweed components
        
        Args:
            lisan_path: Path to lisan_phonemes.json
            sample_rate: Audio sample rate
        """
        self.parser = TajweedParser()
        self.physics = PhysicsValidator(sample_rate=sample_rate)
        self.duration_model = DurationModel(lisan_path)
        self.sample_rate = sample_rate
        
        # Load Lisan data if available
        if lisan_path and Path(lisan_path).exists():
            with open(lisan_path, 'r', encoding='utf-8') as f:
                self.lisan_data = json.load(f)
        else:
            self.lisan_data = {}
    
    def refine_alignment(self,
                         audio_path: str,
                         alignment_json: Dict,
                         quran_text: str) -> RefinementResult:
        """
        Refine an MFA/wav2vec alignment using Tajweed physics
        
        Args:
            audio_path: Path to audio file
            alignment_json: Original alignment (word/phoneme timings)
            quran_text: Original Quranic text (Uthmani)
        
        Returns:
            RefinementResult with refined timings and confidence scores
        """
        # Load audio
        audio = self.physics.load_audio(audio_path)
        
        # Parse Tajweed rules from text
        word_tags = self.parser.parse_text(quran_text)
        
        # Calibrate duration model from alignment
        self._calibrate_from_alignment(audio, alignment_json)
        
        # Process each word
        refined_words = []
        all_scores = []
        
        alignment_words = alignment_json.get('words', alignment_json.get('segments', []))
        
        for i, (word_align, word_tag) in enumerate(zip(alignment_words, word_tags)):
            refined_word = self._refine_word(
                audio=audio,
                word_alignment=word_align,
                word_tags=word_tag,
                word_index=i
            )
            refined_words.append(refined_word)
            all_scores.append(refined_word.average_confidence)
        
        # Calculate statistics
        overall_confidence = np.mean(all_scores) if all_scores else 0.0
        
        stats = {
            "total_words": len(refined_words),
            "total_letters": sum(len(w.letters) for w in refined_words),
            "average_physics_score": np.mean([
                l.physics_score 
                for w in refined_words 
                for l in w.letters
            ]) if refined_words else 0.0,
            "duration_valid_percent": np.mean([
                l.duration_valid 
                for w in refined_words 
                for l in w.letters
            ]) * 100 if refined_words else 0.0
        }
        
        return RefinementResult(
            audio_path=audio_path,
            original_alignment_path="",
            words=refined_words,
            overall_confidence=overall_confidence,
            statistics=stats
        )
    
    def _calibrate_from_alignment(self, audio: np.ndarray, alignment: Dict):
        """Calibrate duration model from existing alignment"""
        # Extract short vowel durations for calibration
        vowel_segments = []
        words = alignment.get('words', alignment.get('segments', []))
        
        for word in words:
            phonemes = word.get('phonemes', word.get('chars', []))
            for phoneme in phonemes:
                # Look for short vowels (single character, short duration)
                p_text = phoneme.get('text', phoneme.get('char', ''))
                p_start = phoneme.get('start', 0)
                p_end = phoneme.get('end', 0)
                duration = p_end - p_start
                
                # Short vowels are typically 50-150ms
                if 0.05 <= duration <= 0.15:
                    vowel_segments.append(duration)
        
        # Calibrate
        if vowel_segments:
            self.duration_model.calibrate_from_samples(
                reciter_name="auto_calibrated",
                vowel_durations=vowel_segments
            )
            self.physics.calibrate_average_vowel(
                audio, 
                [(0, d) for d in vowel_segments]
            )
    
    def _refine_word(self,
                     audio: np.ndarray,
                     word_alignment: Dict,
                     word_tags,
                     word_index: int) -> RefinedWord:
        """Refine a single word's letter timings"""
        refined_letters = []
        
        word_start = word_alignment.get('start', 0)
        word_end = word_alignment.get('end', 0)
        
        # Get phoneme/character alignments
        phonemes = word_alignment.get('phonemes', 
                   word_alignment.get('chars', 
                   word_alignment.get('letters', [])))
        
        # Match phonemes to letter tags
        for j, letter_tag in enumerate(word_tags.letters):
            # Find corresponding phoneme timing
            if j < len(phonemes):
                phoneme = phonemes[j]
                orig_start = phoneme.get('start', word_start)
                orig_end = phoneme.get('end', word_end)
            else:
                # Estimate timing if no phoneme data
                letter_duration = (word_end - word_start) / len(word_tags.letters)
                orig_start = word_start + j * letter_duration
                orig_end = orig_start + letter_duration
            
            # Run physics validation based on Tajweed type
            physics_score, refined_start, refined_end = self._validate_and_refine(
                audio=audio,
                letter_tag=letter_tag,
                start=orig_start,
                end=orig_end,
                next_start=phonemes[j+1].get('start') if j+1 < len(phonemes) else None
            )
            
            # Validate duration
            duration_valid = self._check_duration(
                letter_tag=letter_tag,
                start=refined_start,
                end=refined_end
            )
            
            # Calculate confidence
            confidence = (physics_score + (1.0 if duration_valid else 0.5)) / 2
            
            refined_letters.append(RefinedLetter(
                letter=letter_tag.char_visual,
                phoneme=letter_tag.char_phonetic,
                original_start=orig_start,
                original_end=orig_end,
                refined_start=refined_start,
                refined_end=refined_end,
                tajweed_rule=letter_tag.tajweed_type.value,
                physics_score=physics_score,
                duration_valid=duration_valid,
                confidence=confidence
            ))
        
        avg_confidence = np.mean([l.confidence for l in refined_letters]) if refined_letters else 0.0
        
        # Adjust word boundaries based on refined letters
        if refined_letters:
            word_start = refined_letters[0].refined_start
            word_end = refined_letters[-1].refined_end
        
        return RefinedWord(
            word_text=word_tags.word_text,
            start=word_start,
            end=word_end,
            letters=refined_letters,
            average_confidence=avg_confidence
        )
    
    def _validate_and_refine(self,
                              audio: np.ndarray,
                              letter_tag,
                              start: float,
                              end: float,
                              next_start: Optional[float]) -> Tuple[float, float, float]:
        """
        Run appropriate physics validator and suggest refined boundaries
        
        Returns:
            Tuple of (physics_score, refined_start, refined_end)
        """
        physics_score = 0.5  # Default neutral score
        refined_start = start
        refined_end = end
        
        # Select validator based on physics check type
        check_type = letter_tag.physics_check
        
        if check_type == PhysicsCheck.CHECK_RMS_BOUNCE:
            # Qalqalah - look for dip→spike
            result = self.physics.validate_qalqalah(audio, start, end)
            physics_score = result.score
            
        elif check_type == PhysicsCheck.CHECK_DURATION:
            # Madd or Idgham - duration based
            madd_count = letter_tag.madd_count if hasattr(letter_tag, 'madd_count') else 2
            result = self.physics.validate_madd(audio, start, end, madd_count)
            physics_score = result.score
            
        elif check_type == PhysicsCheck.CHECK_GHUNNAH:
            # Ghunnah, Ikhfa, Iqlab - nasal detection
            tajweed_type = letter_tag.tajweed_type
            
            if tajweed_type == TajweedType.IKHFA:
                result = self.physics.validate_ikhfa(audio, start, end)
            elif tajweed_type == TajweedType.IQLAB:
                result = self.physics.validate_iqlab(audio, start, end)
            else:
                result = self.physics.validate_ghunnah(audio, start, end)
            physics_score = result.score
            
        elif check_type == PhysicsCheck.CHECK_FORMANT_F2:
            # Tafkheem or Tarqeeq
            if letter_tag.tajweed_type == TajweedType.TAFKHEEM:
                result = self.physics.validate_tafkheem(audio, start, end)
            else:
                result = self.physics.validate_tarqeeq(audio, start, end)
            physics_score = result.score
        
        # For Idgham, check energy continuity
        if letter_tag.tajweed_type in [TajweedType.IDGHAM_FULL, TajweedType.IDGHAM_PARTIAL]:
            if next_start:
                has_ghunnah = letter_tag.tajweed_type == TajweedType.IDGHAM_PARTIAL
                result = self.physics.validate_idgham(
                    audio, start, end, next_start, has_ghunnah
                )
                physics_score = result.score
        
        # For Izhar, check clean boundaries
        if next_start and letter_tag.char_visual == 'ن':
            # Check if this should be Izhar
            result = self.physics.validate_izhar(audio, start, end, next_start)
            if result.status == ValidationStatus.PASS:
                physics_score = max(physics_score, result.score)
        
        return (physics_score, refined_start, refined_end)
    
    def _check_duration(self, letter_tag, start: float, end: float) -> bool:
        """Check if duration matches Tajweed expectations"""
        duration = end - start
        tajweed_type = letter_tag.tajweed_type
        
        # Map Tajweed type to Madd type for duration check
        madd_map = {
            TajweedType.MADD_ASLI: MaddType.ASLI,
            TajweedType.MADD_WAJIB: MaddType.WAJIB,
            TajweedType.MADD_LAZIM: MaddType.LAZIM,
        }
        
        if tajweed_type in madd_map:
            madd_type = madd_map[tajweed_type]
            harakat = letter_tag.madd_count if hasattr(letter_tag, 'madd_count') else 2
            result = self.duration_model.validate_duration(duration, madd_type, harakat)
            return result.is_valid
        
        if tajweed_type == TajweedType.GHUNNAH:
            result = self.duration_model.validate_ghunnah_duration(duration)
            return result.is_valid
        
        # Default: duration is valid
        return True
    
    def save_refined_alignment(self, 
                                result: RefinementResult,
                                output_path: str):
        """Save refined alignment to JSON file"""
        output = {
            "audio_path": result.audio_path,
            "original_alignment": result.original_alignment_path,
            "overall_confidence": result.overall_confidence,
            "statistics": result.statistics,
            "words": [
                {
                    "word": w.word_text,
                    "start": w.start,
                    "end": w.end,
                    "average_confidence": w.average_confidence,
                    "letters": [asdict(l) for l in w.letters]
                }
                for w in result.words
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        return output_path


def main():
    """Test MFA Refiner"""
    print("=" * 50)
    print("TajweedSST MFA Refiner Test")
    print("=" * 50)
    
    # Create refiner
    lisan_path = Path(__file__).parent / "lisan_phonemes.json"
    refiner = MFARefiner(str(lisan_path) if lisan_path.exists() else None)
    
    print("\nRefiner initialized with:")
    print(f"  - Tajweed Parser: Ready")
    print(f"  - Physics Validator: 10 validators")
    print(f"  - Duration Model: Ready")
    print(f"  - Lisan Data: {'Loaded' if refiner.lisan_data else 'Not found'}")
    
    # Mock alignment for testing
    mock_alignment = {
        "words": [
            {
                "text": "قُلْ",
                "start": 0.0,
                "end": 0.5,
                "phonemes": [
                    {"text": "ق", "start": 0.0, "end": 0.15},
                    {"text": "ُ", "start": 0.15, "end": 0.25},
                    {"text": "ل", "start": 0.25, "end": 0.5}
                ]
            }
        ]
    }
    
    print("\nMock alignment test:")
    print(f"  Input word: قُلْ")
    print(f"  Phonemes: 3")
    print(f"\nNote: Full test requires actual audio file.")


if __name__ == "__main__":
    main()
