#!/usr/bin/env python3
"""
TajweedSST - Main Pipeline Orchestrator

Execution Order:
1. Text Parse: Generate Phonetic Script & Rule Tags
2. WhisperX: Get Word Timestamps
3. MFA: Get Phoneme Timestamps inside Words
4. Math: Clamp/Normalize Phonemes to Words
5. DSP: Run Physics checks on specific tagged timestamps
6. Export: Save JSON
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from .tajweed_parser import TajweedParser, TajweedType, PhysicsCheck, WordTags
from .alignment_engine import AlignmentEngine, MockAlignmentEngine, AlignmentResult
from .physics_validator import PhysicsValidator, ValidationStatus


@dataclass
class PhonemeOutput:
    """Output format for a single phoneme"""
    char_visual: str
    char_phonetic: str
    start: float
    end: float
    tajweed_type: str
    physics_analysis: Optional[Dict] = None
    score: float = 1.0

@dataclass
class WordOutput:
    """Output format for a single word"""
    word_text: str
    whisper_anchor: Dict
    phonemes: List[Dict]

@dataclass
class AyahOutput:
    """Output format for a complete ayah"""
    surah: int
    ayah: int
    words: List[Dict]
    metadata: Dict


class TajweedPipeline:
    """
    Main orchestrator for the TajweedSST pipeline
    """
    
    def __init__(self, 
                 use_mock_alignment: bool = True,
                 device: str = "cuda"):
        """
        Initialize pipeline
        
        Args:
            use_mock_alignment: Use mock alignment for testing (no WhisperX/MFA)
            device: cuda or cpu
        """
        self.parser = TajweedParser()
        
        if use_mock_alignment:
            self.aligner = MockAlignmentEngine()
        else:
            self.aligner = AlignmentEngine(device=device)
        
        self.validator = PhysicsValidator()
        self.use_mock = use_mock_alignment
    
    def process(self,
                audio_path: str,
                text: str,
                surah: int,
                ayah: int) -> Dict:
        """
        Process a single ayah through the complete pipeline
        
        Args:
            audio_path: Path to audio file
            text: Uthmani Quran text for the ayah
            surah: Surah number
            ayah: Ayah number
            
        Returns:
            Complete JSON output with timing and Tajweed analysis
        """
        # Step 1: Parse text and generate Tajweed tags
        word_tags = self.parser.parse_text(text)
        
        # Extract phonetic words for alignment
        phonetic_words = [w.phonetic_stream for w in word_tags]
        
        # Step 2 & 3: Run alignment (WhisperX + MFA)
        alignment = self.aligner.align(
            audio_path=audio_path,
            phonetic_words=phonetic_words,
            surah=surah,
            ayah=ayah
        )
        
        # Step 4: Normalization is done inside alignment_engine
        
        # Step 5: Load audio and run physics validation
        if not self.use_mock:
            audio = self.validator.load_audio(audio_path)
        else:
            import numpy as np
            audio = np.random.randn(22050 * 10) * 0.1  # Mock audio
        
        # Build output
        output_words = []
        
        for word_idx, (word_tag, word_align) in enumerate(zip(word_tags, alignment.words)):
            word_output = {
                "word_text": word_tag.word_text,
                "whisper_anchor": {
                    "start": round(word_align.whisper_start, 3),
                    "end": round(word_align.whisper_end, 3)
                },
                "phonemes": []
            }
            
            # Map phonemes to letters and run physics checks
            for letter_idx, letter_tag in enumerate(word_tag.letters):
                # Skip silent letters
                if letter_tag.is_silent:
                    continue
                
                # Get corresponding phoneme timing
                if letter_idx < len(word_align.phonemes):
                    phoneme_align = word_align.phonemes[letter_idx]
                    start = phoneme_align.start
                    end = phoneme_align.end
                else:
                    # Estimate timing if not aligned
                    word_duration = word_align.whisper_end - word_align.whisper_start
                    num_letters = len([l for l in word_tag.letters if not l.is_silent])
                    letter_duration = word_duration / max(num_letters, 1)
                    start = word_align.whisper_start + (letter_idx * letter_duration)
                    end = start + letter_duration
                
                phoneme_output = {
                    "char_visual": letter_tag.char_visual,
                    "char_phonetic": letter_tag.char_phonetic,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "tajweed_type": letter_tag.tajweed_type.value,
                    "score": 1.0
                }
                
                # Step 5: Run physics validation if tagged
                if letter_tag.physics_check != PhysicsCheck.NONE:
                    physics_result = self._run_physics_check(
                        audio=audio,
                        start=start,
                        end=end,
                        check_type=letter_tag.physics_check,
                        tajweed_type=letter_tag.tajweed_type,
                        madd_count=letter_tag.madd_count
                    )
                    phoneme_output["physics_analysis"] = physics_result
                    phoneme_output["score"] = physics_result.get("score", 1.0)
                
                word_output["phonemes"].append(phoneme_output)
            
            output_words.append(word_output)
        
        # Final output structure
        output = {
            "surah": surah,
            "ayah": ayah,
            "words": output_words,
            "metadata": {
                "audio_path": audio_path,
                "text": text,
                "pipeline_version": "1.0.0",
                "mock_alignment": self.use_mock
            }
        }
        
        return output
    
    def _run_physics_check(self,
                           audio,
                           start: float,
                           end: float,
                           check_type: PhysicsCheck,
                           tajweed_type: TajweedType,
                           madd_count: int = 0) -> Dict:
        """Run appropriate physics check based on tag"""
        
        if check_type == PhysicsCheck.CHECK_RMS_BOUNCE:
            result = self.validator.validate_qalqalah(audio, start, end)
            return {
                "check_type": "Qalqalah_RMS",
                "rms_profile": result.rms_profile,
                "dip_depth": round(result.dip_depth, 3),
                "spike_height": round(result.spike_height, 3),
                "status": result.status.value,
                "score": round(result.score, 3)
            }
        
        elif check_type == PhysicsCheck.CHECK_DURATION:
            result = self.validator.validate_madd(audio, start, end, madd_count or 2)
            return {
                "check_type": "Madd_Duration",
                "actual_duration_ms": round(result.actual_duration_ms, 1),
                "expected_duration_ms": round(result.expected_duration_ms, 1),
                "ratio": round(result.ratio, 2),
                "status": result.status.value,
                "score": round(result.score, 3)
            }
        
        elif check_type == PhysicsCheck.CHECK_GHUNNAH:
            result = self.validator.validate_ghunnah(audio, start, end)
            return {
                "check_type": "Ghunnah_Formant",
                "nasal_detected": result.nasal_formant_detected,
                "pitch_stability": round(result.pitch_stability, 3),
                "duration_elongation": round(result.duration_elongation, 2),
                "status": result.status.value,
                "score": round(result.score, 3)
            }
        
        elif check_type == PhysicsCheck.CHECK_FORMANT_F2:
            result = self.validator.validate_tafkheem(audio, start, end)
            return {
                "check_type": "Tafkheem_F2",
                "f2_value_hz": round(result.f2_value_hz, 0),
                "depression_ratio": round(result.depression_ratio, 3),
                "status": result.status.value,
                "score": round(result.score, 3)
            }
        
        return {"check_type": "None", "status": "SKIPPED", "score": 1.0}
    
    def process_batch(self,
                      audio_dir: str,
                      quran_json_path: str,
                      output_dir: str,
                      surah: int,
                      start_ayah: int = 1,
                      end_ayah: Optional[int] = None) -> List[str]:
        """
        Process multiple ayahs in batch
        
        Args:
            audio_dir: Directory containing audio files (named {surah}_{ayah}.mp3)
            quran_json_path: Path to Quran text JSON
            output_dir: Directory to save output JSON files
            surah: Surah to process
            start_ayah: Starting ayah number
            end_ayah: Ending ayah number (None = all)
            
        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Quran text
        with open(quran_json_path, 'r', encoding='utf-8') as f:
            quran_data = json.load(f)
        
        output_files = []
        
        # Process each ayah
        for ayah in range(start_ayah, (end_ayah or len(quran_data.get(str(surah), []))) + 1):
            audio_path = Path(audio_dir) / f"{surah}_{ayah}.mp3"
            
            if not audio_path.exists():
                print(f"Skipping {surah}:{ayah} - audio not found")
                continue
            
            # Get text
            text = quran_data.get(str(surah), {}).get(str(ayah), "")
            if not text:
                print(f"Skipping {surah}:{ayah} - text not found")
                continue
            
            # Process
            result = self.process(
                audio_path=str(audio_path),
                text=text,
                surah=surah,
                ayah=ayah
            )
            
            # Save
            output_path = output_dir / f"{surah}_{ayah}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            
            output_files.append(str(output_path))
            print(f"Processed {surah}:{ayah} → {output_path}")
        
        return output_files


def main():
    """Demo the pipeline"""
    print("=" * 60)
    print("TajweedSST Pipeline Demo")
    print("=" * 60)
    
    pipeline = TajweedPipeline(use_mock_alignment=True)
    
    # Test with Surah Al-Ikhlas, Ayah 1
    test_text = "قُلْ هُوَ اللَّهُ أَحَدٌ"
    
    print(f"\nInput Text: {test_text}")
    print("\nProcessing...")
    
    result = pipeline.process(
        audio_path="test_audio.mp3",
        text=test_text,
        surah=112,
        ayah=1
    )
    
    print("\n" + "=" * 60)
    print("OUTPUT JSON:")
    print("=" * 60)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
