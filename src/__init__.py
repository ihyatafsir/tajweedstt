"""
TajweedSST - Quranic Precision Alignment & Tajweed Analysis Tool

A Python-based pipeline that generates letter-level precise timing data
for Quran recitations, prevents timing drift, and uses signal processing
to validate Tajweed rules.

Usage:
    from tajweedsst.src.pipeline import TajweedPipeline
    
    pipeline = TajweedPipeline()
    result = pipeline.process(
        audio_path="path/to/audio.mp3",
        text="قُلْ هُوَ اللَّهُ أَحَدٌ",
        surah=112,
        ayah=1
    )
"""

from .tajweed_parser import TajweedParser, TajweedType, PhysicsCheck
from .alignment_engine import AlignmentEngine, MockAlignmentEngine
from .physics_validator import PhysicsValidator, ValidationStatus
from .pipeline import TajweedPipeline

__version__ = "1.0.0"
__all__ = [
    "TajweedPipeline",
    "TajweedParser",
    "TajweedType",
    "PhysicsCheck",
    "AlignmentEngine",
    "MockAlignmentEngine",
    "PhysicsValidator",
    "ValidationStatus"
]
