#!/usr/bin/env python3
"""
TajweedSST - Step 1: Tajweed Rule Parser

Generates two parallel text streams and a Rule Map:
- Visual Stream: Standard Uthmani text
- Phonetic Stream: Pronounced text for MFA
- Tajweed Map: Tags for physics validation

Tajweed Rules Implemented:
- Idgham (Assimilation)
- Iqlab (Conversion)
- Ikhfa (Concealment)
- Qalqalah (Bounce)
- Ghunnah (Nasalization)
- Madd (Elongation)
- Tafkheem/Tarqeeq (Heavy/Light)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum

class TajweedType(Enum):
    NONE = "None"
    QALQALAH_SUGHRA = "Qalqalah_Sughra"
    QALQALAH_KUBRA = "Qalqalah_Kubra"
    GHUNNAH = "Ghunnah"
    IDGHAM_FULL = "Idgham_Full"
    IDGHAM_PARTIAL = "Idgham_Partial"
    IQLAB = "Iqlab"
    IKHFA = "Ikhfa"
    MADD_ASLI = "Madd_Asli"
    MADD_WAJIB = "Madd_Wajib"
    MADD_LAZIM = "Madd_Lazim"
    TAFKHEEM = "Tafkheem"
    TARQEEQ = "Tarqeeq"
    SILENT = "Silent"

class PhysicsCheck(Enum):
    CHECK_RMS_BOUNCE = "Check_RMS_Bounce"
    CHECK_DURATION = "Check_Duration"
    CHECK_GHUNNAH = "Check_Ghunnah"
    CHECK_FORMANT_F2 = "Check_Formant_F2"
    NONE = "None"

@dataclass
class LetterTag:
    """Tag for a single Arabic letter with Tajweed info"""
    char_visual: str
    char_phonetic: str
    position: int
    tajweed_type: TajweedType = TajweedType.NONE
    physics_check: PhysicsCheck = PhysicsCheck.NONE
    is_silent: bool = False
    madd_count: int = 0  # 0=none, 2=asli, 4=wajib, 6=lazim

@dataclass
class WordTags:
    """Tajweed tags for a complete word"""
    word_text: str
    letters: List[LetterTag] = field(default_factory=list)
    phonetic_stream: str = ""

class TajweedParser:
    """Parses Uthmani Quran text and generates Tajweed rule tags"""
    
    # Qalqalah letters: ق ط ب ج د
    QALQALAH_LETTERS = set('قطبجد')
    
    # Heavy letters (Tafkheem): خ ص ض غ ط ق ظ
    TAFKHEEM_LETTERS = set('خصضغطقظ')
    
    # Idgham letters after Nun Sakinah: ي ر م ل و ن
    IDGHAM_LETTERS = set('يرملون')
    IDGHAM_WITH_GHUNNAH = set('ينمو')  # With Ghunnah
    IDGHAM_WITHOUT_GHUNNAH = set('رل')  # Without Ghunnah
    
    # Ikhfa letters (15 letters)
    IKHFA_LETTERS = set('تثجدذزسشصضطظفقك')
    
    # Harakat (vowel marks)
    FATHA = '\u064E'
    DAMMA = '\u064F'
    KASRA = '\u0650'
    SUKUN = '\u0652'
    SHADDA = '\u0651'
    TANWEEN_FATH = '\u064B'
    TANWEEN_DAMM = '\u064C'
    TANWEEN_KASR = '\u064D'
    
    # Madd letters
    MADD_ALIF = 'ا'
    MADD_WAW = 'و'
    MADD_YA = 'ي'
    
    # Phonetic mapping (simplified Buckwalter-like)
    PHONETIC_MAP = {
        'ا': 'ā', 'ب': 'b', 'ت': 't', 'ث': 'ṯ', 'ج': 'j', 'ح': 'ḥ',
        'خ': 'ḫ', 'د': 'd', 'ذ': 'ḏ', 'ر': 'r', 'ز': 'z', 'س': 's',
        'ش': 'š', 'ص': 'ṣ', 'ض': 'ḍ', 'ط': 'ṭ', 'ظ': 'ẓ', 'ع': 'ʿ',
        'غ': 'ġ', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm',
        'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'y', 'ء': 'ʾ', 'ة': 'h',
        'ى': 'ā', 'ئ': 'ʾ', 'ؤ': 'ʾ', 'أ': 'ʾa', 'إ': 'ʾi', 'آ': 'ʾā'
    }
    
    def __init__(self):
        self.debug = False
    
    def parse_text(self, text: str) -> List[WordTags]:
        """Parse Uthmani text and return tagged words"""
        words = text.strip().split()
        result = []
        
        for word in words:
            word_tags = self._parse_word(word)
            result.append(word_tags)
        
        # Cross-word analysis (Nun Sakinah rules across words)
        self._analyze_cross_word_rules(result)
        
        return result
    
    def _parse_word(self, word: str) -> WordTags:
        """Parse a single word and generate letter tags"""
        word_tags = WordTags(word_text=word)
        
        # Extract base letters and diacritics
        letters_with_harakat = self._split_letters(word)
        
        for idx, (letter, harakat) in enumerate(letters_with_harakat):
            tag = self._analyze_letter(
                letter=letter,
                harakat=harakat,
                position=idx,
                context=(letters_with_harakat, idx),
                word=word
            )
            word_tags.letters.append(tag)
        
        # Generate phonetic stream
        word_tags.phonetic_stream = self._generate_phonetic_stream(word_tags.letters)
        
        return word_tags
    
    def _split_letters(self, word: str) -> List[Tuple[str, str]]:
        """Split word into (letter, harakat) pairs"""
        result = []
        i = 0
        harakat_chars = set([self.FATHA, self.DAMMA, self.KASRA, self.SUKUN,
                            self.SHADDA, self.TANWEEN_FATH, self.TANWEEN_DAMM, 
                            self.TANWEEN_KASR, '\u0653', '\u0654', '\u0655',
                            '\u0656', '\u0657', '\u0658', '\u065C', '\u0670'])
        
        while i < len(word):
            char = word[i]
            
            # Skip if it's a harakat
            if char in harakat_chars:
                i += 1
                continue
            
            # Collect harakat following this letter
            harakat = ""
            j = i + 1
            while j < len(word) and word[j] in harakat_chars:
                harakat += word[j]
                j += 1
            
            result.append((char, harakat))
            i = j
        
        return result
    
    def _analyze_letter(self, letter: str, harakat: str, position: int,
                       context: Tuple[List, int], word: str) -> LetterTag:
        """Analyze a single letter and assign Tajweed rules"""
        letters_list, idx = context
        is_last = idx == len(letters_list) - 1
        has_sukun = self.SUKUN in harakat
        has_shadda = self.SHADDA in harakat
        
        tag = LetterTag(
            char_visual=letter,
            char_phonetic=self.PHONETIC_MAP.get(letter, letter),
            position=position
        )
        
        # Rule 1: Qalqalah (ق ط ب ج د with Sukun)
        if letter in self.QALQALAH_LETTERS and (has_sukun or is_last):
            if is_last:
                tag.tajweed_type = TajweedType.QALQALAH_KUBRA
            else:
                tag.tajweed_type = TajweedType.QALQALAH_SUGHRA
            tag.physics_check = PhysicsCheck.CHECK_RMS_BOUNCE
        
        # Rule 2: Tafkheem (Heavy letters)
        elif letter in self.TAFKHEEM_LETTERS:
            tag.tajweed_type = TajweedType.TAFKHEEM
            tag.physics_check = PhysicsCheck.CHECK_FORMANT_F2
        
        # Rule 3: Madd (Elongation) - check preceding vowel
        elif letter in [self.MADD_ALIF, self.MADD_WAW, self.MADD_YA]:
            # Check for Madd conditions
            if idx > 0:
                prev_letter, prev_harakat = letters_list[idx - 1]
                if (letter == self.MADD_ALIF and self.FATHA in prev_harakat) or \
                   (letter == self.MADD_WAW and self.DAMMA in prev_harakat) or \
                   (letter == self.MADD_YA and self.KASRA in prev_harakat):
                    # Check what follows for Madd type
                    if is_last:
                        tag.tajweed_type = TajweedType.MADD_ASLI
                        tag.madd_count = 2
                    elif idx + 1 < len(letters_list):
                        next_letter, next_harakat = letters_list[idx + 1]
                        if self.SHADDA in next_harakat or self.SUKUN in next_harakat:
                            tag.tajweed_type = TajweedType.MADD_LAZIM
                            tag.madd_count = 6
                        else:
                            tag.tajweed_type = TajweedType.MADD_WAJIB
                            tag.madd_count = 4
                    tag.physics_check = PhysicsCheck.CHECK_DURATION
        
        # Rule 4: Ghunnah (Nun/Meem with Shadda)
        if letter in 'نم' and has_shadda:
            tag.tajweed_type = TajweedType.GHUNNAH
            tag.physics_check = PhysicsCheck.CHECK_GHUNNAH
        
        # Rule 5: Nun Sakinah / Tanween rules
        if letter == 'ن' and has_sukun:
            if idx + 1 < len(letters_list):
                next_letter, _ = letters_list[idx + 1]
                # Iqlab: Nun + Ba → Mim + Ba
                if next_letter == 'ب':
                    tag.tajweed_type = TajweedType.IQLAB
                    tag.char_phonetic = 'm'  # Pronounced as Mim
                    tag.physics_check = PhysicsCheck.CHECK_GHUNNAH
                # Idgham
                elif next_letter in self.IDGHAM_LETTERS:
                    if next_letter in self.IDGHAM_WITH_GHUNNAH:
                        tag.tajweed_type = TajweedType.IDGHAM_PARTIAL
                    else:
                        tag.tajweed_type = TajweedType.IDGHAM_FULL
                    tag.physics_check = PhysicsCheck.CHECK_DURATION
                # Ikhfa
                elif next_letter in self.IKHFA_LETTERS:
                    tag.tajweed_type = TajweedType.IKHFA
                    tag.physics_check = PhysicsCheck.CHECK_GHUNNAH
        
        # Handle Tanween similarly
        if any(tanween in harakat for tanween in [self.TANWEEN_FATH, self.TANWEEN_DAMM, self.TANWEEN_KASR]):
            if idx + 1 < len(letters_list):
                next_letter, _ = letters_list[idx + 1]
                if next_letter == 'ب':
                    tag.tajweed_type = TajweedType.IQLAB
                    tag.physics_check = PhysicsCheck.CHECK_GHUNNAH
                elif next_letter in self.IKHFA_LETTERS:
                    tag.tajweed_type = TajweedType.IKHFA
                    tag.physics_check = PhysicsCheck.CHECK_GHUNNAH
        
        # Silent letters (Alif after Waw al-Jama'a, etc.)
        if letter == 'ا' and not harakat and idx > 0:
            prev_letter, prev_harakat = letters_list[idx - 1]
            if prev_letter == 'و' and (self.DAMMA in prev_harakat or self.SUKUN in prev_harakat):
                tag.is_silent = True
                tag.tajweed_type = TajweedType.SILENT
                tag.char_phonetic = ''
        
        return tag
    
    def _analyze_cross_word_rules(self, words: List[WordTags]) -> None:
        """Analyze Tajweed rules that span word boundaries"""
        for i in range(len(words) - 1):
            current_word = words[i]
            next_word = words[i + 1]
            
            if not current_word.letters or not next_word.letters:
                continue
            
            last_letter = current_word.letters[-1]
            first_of_next = next_word.letters[0]
            
            # Check Nun Sakinah at end of word + next word's first letter
            if last_letter.char_visual == 'ن' and last_letter.tajweed_type == TajweedType.NONE:
                if first_of_next.char_visual == 'ب':
                    last_letter.tajweed_type = TajweedType.IQLAB
                    last_letter.char_phonetic = 'm'
                    last_letter.physics_check = PhysicsCheck.CHECK_GHUNNAH
                elif first_of_next.char_visual in self.IDGHAM_LETTERS:
                    if first_of_next.char_visual in self.IDGHAM_WITH_GHUNNAH:
                        last_letter.tajweed_type = TajweedType.IDGHAM_PARTIAL
                    else:
                        last_letter.tajweed_type = TajweedType.IDGHAM_FULL
                    last_letter.physics_check = PhysicsCheck.CHECK_DURATION
                elif first_of_next.char_visual in self.IKHFA_LETTERS:
                    last_letter.tajweed_type = TajweedType.IKHFA
                    last_letter.physics_check = PhysicsCheck.CHECK_GHUNNAH
    
    def _generate_phonetic_stream(self, letters: List[LetterTag]) -> str:
        """Generate phonetic transcription for MFA"""
        phonemes = []
        for letter in letters:
            if not letter.is_silent and letter.char_phonetic:
                phonemes.append(letter.char_phonetic)
        return ' '.join(phonemes)


def main():
    """Test the Tajweed parser"""
    parser = TajweedParser()
    
    # Test with Surah Al-Ikhlas
    test_text = "قُلْ هُوَ اللَّهُ أَحَدٌ"
    
    print("=" * 50)
    print("TajweedSST Parser Test")
    print("=" * 50)
    print(f"Input: {test_text}")
    print()
    
    words = parser.parse_text(test_text)
    
    for word in words:
        print(f"Word: {word.word_text}")
        print(f"  Phonetic: {word.phonetic_stream}")
        for letter in word.letters:
            if letter.tajweed_type != TajweedType.NONE:
                print(f"  [{letter.char_visual}] → {letter.tajweed_type.value} ({letter.physics_check.value})")
        print()


if __name__ == "__main__":
    main()
