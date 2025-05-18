#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for the data loading module.
"""

import os
import sys
import pytest
from pathlib import Path

# Add the project root to sys.path to import the module
project_root = Path(__file__).parents[1]
sys.path.insert(0, str(project_root))

from src.data.load_data import create_directories, RAW_DATA_DIR


def test_create_directories():
    """Test the create_directories function."""
    # Remove the directory if it exists
    if os.path.exists(RAW_DATA_DIR):
        os.rmdir(RAW_DATA_DIR)
    
    # Call the function to create directories
    create_directories()
    
    # Check if the directory was created
    assert os.path.exists(RAW_DATA_DIR)
    assert os.path.isdir(RAW_DATA_DIR) 
