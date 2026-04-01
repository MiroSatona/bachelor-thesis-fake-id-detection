"""
Author: Martin Zůbek
Date: 2026-03-28
File: enums.py
Description: This file contains the enums for the Extractor classes.
"""
from enum import Enum


class DeviceType(str,Enum):
    CPU = "cpu"
    CUDA = "cuda"