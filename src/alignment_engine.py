#!/usr/bin/env python3
"""
TajweedSST - Step 2: Hierarchical Alignment Engine

The Anti-Drift Engine:
1. WhisperX: Get word-level anchors (rigid boundaries)
2. MFA: Get phoneme-level precision within words
3. Normalization: Clamp MFA durations to match WhisperX exactly

Formula: Phoneme_New_Duration = Phoneme_Old * (Whisper_Word_Duration / Sum_MFA_Phonemes)
"""

import os
import json
import subprocess
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path

@dataclass
class PhonemeAlignment:
    """Single phoneme timing"""
    phoneme: str
    start: float
    end: float
    duration: float
    
    @property
    def normalized_duration(self) -> float:
        return self.end - self.start

@dataclass  
class WordAlignment:
    """Word-level alignment with phoneme breakdown"""
    word_text: str
    whisper_start: float
    whisper_end: float
    phonemes: List[PhonemeAlignment] = field(default_factory=list)
    
    @property
    def whisper_duration(self) -> float:
        return self.whisper_end - self.whisper_start

@dataclass
class AlignmentResult:
    """Complete alignment for an audio segment"""
    audio_path: str
    surah: int
    ayah: int
    words: List[WordAlignment] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class AlignmentEngine:
    """
    Hierarchical alignment using WhisperX + MFA
    """
    
    def __init__(self, 
                 whisperx_model: str = "large-v3",
                 mfa_acoustic_model: str = "arabic_mfa",
                 mfa_dictionary: str = "arabic_mfa",
                 device: str = "cuda",
                 compute_type: str = "float16"):
        """
        Initialize alignment engine
        
        Args:
            whisperx_model: WhisperX model size
            mfa_acoustic_model: MFA acoustic model for Arabic
            mfa_dictionary: MFA pronunciation dictionary
            device: cuda or cpu
            compute_type: float16 or float32
        """
        self.whisperx_model = whisperx_model
        self.mfa_acoustic_model = mfa_acoustic_model
        self.mfa_dictionary = mfa_dictionary
        self.device = device
        self.compute_type = compute_type
        
        self._whisperx = None
        self._whisperx_align_model = None
        
    def _load_whisperx(self):
        """Lazy load WhisperX models"""
        if self._whisperx is None:
            import whisperx
            self._whisperx = whisperx.load_model(
                self.whisperx_model,
                device=self.device,
                compute_type=self.compute_type
            )
            # Load alignment model for Arabic
            self._whisperx_align_model, self._whisperx_align_metadata = whisperx.load_align_model(
                language_code="ar",
                device=self.device
            )
    
    def align(self, 
              audio_path: str, 
              phonetic_words: List[str],
              surah: int = 0,
              ayah: int = 0) -> AlignmentResult:
        """
        Perform hierarchical alignment
        
        Args:
            audio_path: Path to audio file
            phonetic_words: List of phonetic transcriptions from TajweedParser
            surah: Surah number for metadata
            ayah: Ayah number for metadata
            
        Returns:
            AlignmentResult with word and phoneme timings
        """
        result = AlignmentResult(
            audio_path=audio_path,
            surah=surah,
            ayah=ayah
        )
        
        # Step 1: WhisperX word-level alignment
        whisper_words = self._run_whisperx(audio_path)
        
        # Step 2: MFA phoneme-level alignment for each word
        mfa_phonemes = self._run_mfa(audio_path, phonetic_words)
        
        # Step 3: Normalize MFA phonemes to WhisperX word boundaries
        for i, (whisper_word, phonemes) in enumerate(zip(whisper_words, mfa_phonemes)):
            word_alignment = WordAlignment(
                word_text=whisper_word['word'],
                whisper_start=whisper_word['start'],
                whisper_end=whisper_word['end']
            )
            
            # Normalize phoneme durations
            normalized_phonemes = self._normalize_phonemes(
                phonemes=phonemes,
                target_start=whisper_word['start'],
                target_end=whisper_word['end']
            )
            word_alignment.phonemes = normalized_phonemes
            
            result.words.append(word_alignment)
        
        return result
    
    def _run_whisperx(self, audio_path: str) -> List[Dict]:
        """
        Run WhisperX for word-level timing
        
        Returns: List of {word, start, end} dicts
        """
        self._load_whisperx()
        import whisperx
        
        # Transcribe
        audio = whisperx.load_audio(audio_path)
        result = self._whisperx.transcribe(audio, batch_size=16)
        
        # Align to get word-level timestamps
        aligned = whisperx.align(
            result["segments"],
            self._whisperx_align_model,
            self._whisperx_align_metadata,
            audio,
            self.device,
            return_char_alignments=False
        )
        
        # Extract word timings
        words = []
        for segment in aligned["segments"]:
            for word_data in segment.get("words", []):
                words.append({
                    "word": word_data["word"],
                    "start": word_data["start"],
                    "end": word_data["end"]
                })
        
        return words
    
    def _run_mfa(self, audio_path: str, phonetic_words: List[str]) -> List[List[Dict]]:
        """
        Run MFA for phoneme-level timing within each word
        
        Returns: List of phoneme lists per word
        """
        # Create temp directory for MFA
        temp_dir = Path("/tmp/tajweedsst_mfa")
        temp_dir.mkdir(exist_ok=True)
        
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        
        # Copy audio and create transcript
        audio_name = Path(audio_path).stem
        transcript_path = input_dir / f"{audio_name}.txt"
        
        # Write phonetic transcript (space-separated words)
        transcript = " ".join(phonetic_words)
        transcript_path.write_text(transcript)
        
        # Copy audio file
        import shutil
        audio_dest = input_dir / Path(audio_path).name
        shutil.copy(audio_path, audio_dest)
        
        # Run MFA
        try:
            subprocess.run([
                "mfa", "align",
                str(input_dir),
                self.mfa_dictionary,
                self.mfa_acoustic_model,
                str(output_dir),
                "--clean",
                "--quiet"
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"MFA Error: {e.stderr.decode()}")
            return [[] for _ in phonetic_words]
        
        # Parse TextGrid output
        textgrid_path = output_dir / f"{audio_name}.TextGrid"
        if textgrid_path.exists():
            return self._parse_textgrid(textgrid_path, len(phonetic_words))
        
        return [[] for _ in phonetic_words]
    
    def _parse_textgrid(self, textgrid_path: Path, word_count: int) -> List[List[Dict]]:
        """Parse MFA TextGrid output for phoneme timings"""
        try:
            import textgrid
            tg = textgrid.TextGrid.fromFile(str(textgrid_path))
            
            # Find phones tier
            phones_tier = None
            words_tier = None
            for tier in tg:
                if tier.name == "phones":
                    phones_tier = tier
                elif tier.name == "words":
                    words_tier = tier
            
            if not phones_tier or not words_tier:
                return [[] for _ in range(word_count)]
            
            # Group phonemes by word boundaries
            result = []
            word_idx = 0
            current_word_phones = []
            
            for interval in phones_tier:
                if interval.mark and interval.mark != "":
                    phone_data = {
                        "phoneme": interval.mark,
                        "start": interval.minTime,
                        "end": interval.maxTime
                    }
                    
                    # Check if we've moved to next word
                    if word_idx < len(words_tier):
                        word_interval = words_tier[word_idx]
                        if interval.minTime >= word_interval.maxTime:
                            result.append(current_word_phones)
                            current_word_phones = []
                            word_idx += 1
                    
                    current_word_phones.append(phone_data)
            
            # Don't forget last word
            if current_word_phones:
                result.append(current_word_phones)
            
            return result
            
        except Exception as e:
            print(f"TextGrid parse error: {e}")
            return [[] for _ in range(word_count)]
    
    def _normalize_phonemes(self,
                           phonemes: List[Dict],
                           target_start: float,
                           target_end: float) -> List[PhonemeAlignment]:
        """
        Normalize MFA phonemes to fit exactly within WhisperX word boundaries
        
        Formula: Phoneme_New_Duration = Phoneme_Old * (Whisper_Word_Duration / Sum_MFA_Phonemes)
        """
        if not phonemes:
            return []
        
        target_duration = target_end - target_start
        
        # Calculate total MFA duration
        mfa_total = sum(p['end'] - p['start'] for p in phonemes)
        
        if mfa_total == 0:
            return []
        
        # Scale factor
        scale = target_duration / mfa_total
        
        # Normalize each phoneme
        normalized = []
        current_time = target_start
        
        for phone in phonemes:
            old_duration = phone['end'] - phone['start']
            new_duration = old_duration * scale
            
            normalized.append(PhonemeAlignment(
                phoneme=phone['phoneme'],
                start=current_time,
                end=current_time + new_duration,
                duration=new_duration
            ))
            
            current_time += new_duration
        
        # Ensure last phoneme ends exactly at target_end (floating point fix)
        if normalized:
            normalized[-1].end = target_end
            normalized[-1].duration = target_end - normalized[-1].start
        
        return normalized


class MockAlignmentEngine(AlignmentEngine):
    """
    Mock alignment engine for testing without WhisperX/MFA installed
    """
    
    def align(self, 
              audio_path: str,
              phonetic_words: List[str],
              surah: int = 0,
              ayah: int = 0) -> AlignmentResult:
        """Generate mock alignment data"""
        result = AlignmentResult(
            audio_path=audio_path,
            surah=surah,
            ayah=ayah
        )
        
        # Mock timing: 0.5s per word
        current_time = 0.0
        word_duration = 0.5
        
        for word in phonetic_words:
            phonemes = word.split()
            phoneme_duration = word_duration / max(len(phonemes), 1)
            
            word_alignment = WordAlignment(
                word_text=word,
                whisper_start=current_time,
                whisper_end=current_time + word_duration
            )
            
            phoneme_time = current_time
            for phoneme in phonemes:
                word_alignment.phonemes.append(PhonemeAlignment(
                    phoneme=phoneme,
                    start=phoneme_time,
                    end=phoneme_time + phoneme_duration,
                    duration=phoneme_duration
                ))
                phoneme_time += phoneme_duration
            
            result.words.append(word_alignment)
            current_time += word_duration + 0.1  # Gap between words
        
        return result


def main():
    """Test alignment engine"""
    print("=" * 50)
    print("TajweedSST Alignment Engine Test")
    print("=" * 50)
    
    # Use mock engine for testing
    engine = MockAlignmentEngine()
    
    # Test phonetic words from TajweedParser
    phonetic_words = ["q l", "h w", "ā l l ā h", "ʾ ḥ d"]
    
    result = engine.align(
        audio_path="test.wav",
        phonetic_words=phonetic_words,
        surah=112,
        ayah=1
    )
    
    print(f"Aligned {len(result.words)} words:")
    for word in result.words:
        print(f"\n  Word: '{word.word_text}'")
        print(f"  Anchor: {word.whisper_start:.3f} - {word.whisper_end:.3f}s")
        for phoneme in word.phonemes:
            print(f"    [{phoneme.phoneme}] {phoneme.start:.3f} - {phoneme.end:.3f}s")


if __name__ == "__main__":
    main()
