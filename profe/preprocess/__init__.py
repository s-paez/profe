"""
PROFE's preprocess module.

This module:
    - Move FITS files in the `data/` folder to `organized_data/` folder
    - Apply median filter with 3x3-pixel window size
"""

from .fits_processor import FitsProcessor
from .median_filter import MedianFilter

__all__: list[str] = ["FitsProcessor", "MedianFilter"]
