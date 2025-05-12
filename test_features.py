#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the generate_features function to ensure it correctly handles all edge cases.
"""

import os
import sys
import numpy as np
from datetime import datetime

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import generate_features function from the API module
try:
    from src.api.main import generate_features
    print("Successfully imported generate_features function.")
except ImportError as e:
    print(f"Error importing generate_features: {e}")
    print("Make sure the API module is properly installed.")
    sys.exit(1)

def test_store_features():
    """Test store features with different store numbers."""
    print("\n=== Testing Store Features ===")
    date = datetime.now()
    
    # Test with valid store numbers
    for store_nbr in [1, 10, 25, 54]:
        features = generate_features(store_nbr, "PRODUCE", False, date)
        print(f"Store {store_nbr}: Generated {len(features)} features")
        
        # Check that the proper store feature is set
        store_idx = 8 + (store_nbr - 1)
        if features[store_idx] == 1:
            print(f"✅ Store {store_nbr} correctly set at index {store_idx}")
        else:
            print(f"❌ Store {store_nbr} not set correctly at index {store_idx}")
    
    # Test with invalid store numbers
    for store_nbr in [0, 55, 100]:
        features = generate_features(store_nbr, "PRODUCE", False, date)
        print(f"Invalid store {store_nbr}: Generated {len(features)} features (should be all zeros for store indices)")
        
        # Check that no store feature is set (no index errors)
        store_indices = list(range(8, 62))
        if not any(features[i] == 1 for i in store_indices):
            print(f"✅ Invalid store {store_nbr} handled correctly")
        else:
            print(f"❌ Some store feature was incorrectly set for invalid store {store_nbr}")

def test_family_features():
    """Test family features with different family names."""
    print("\n=== Testing Family Features ===")
    date = datetime.now()
    
    # Test with valid family names
    valid_families = [
        'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 
        'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 
        'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE',
        'HOME AND KITCHEN', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR'
    ]
    
    for i, family in enumerate(valid_families):
        features = generate_features(1, family, False, date)
        print(f"Family {family}: Generated {len(features)} features")
        
        # Check that the proper family feature is set
        family_idx = 62 + i
        if features[family_idx] == 1:
            print(f"✅ Family {family} correctly set at index {family_idx}")
        else:
            print(f"❌ Family {family} not set correctly at index {family_idx}")
    
    # Test with invalid family names
    invalid_families = ['INVALID', 'NOT_A_FAMILY', 'TEST']
    for family in invalid_families:
        features = generate_features(1, family, False, date)
        print(f"Invalid family {family}: Generated {len(features)} features (should be all zeros for family indices)")
        
        # Check that no family feature is set
        family_indices = list(range(62, 81))
        if not any(features[i] == 1 for i in family_indices):
            print(f"✅ Invalid family {family} handled correctly")
        else:
            print(f"❌ Some family feature was incorrectly set for invalid family {family}")

def test_date_features():
    """Test date features with different date values."""
    print("\n=== Testing Date Features ===")
    
    # Test with valid date
    valid_date = datetime(2025, 5, 10)
    features = generate_features(1, "PRODUCE", False, valid_date)
    
    # Check date features
    expected_date_features = [
        (1, 2025),      # year
        (2, 5),         # month
        (3, 10),        # day
        (4, 5),         # dayofweek (Saturday is 5)
        (5, 130),       # dayofyear
        (6, 2),         # quarter
        (7, 1)          # is_weekend
    ]
    
    print("Checking date features:")
    for idx, expected in expected_date_features:
        if features[idx] == expected:
            print(f"✅ Date feature at index {idx} correctly set to {expected}")
        else:
            print(f"❌ Date feature at index {idx} set to {features[idx]}, expected {expected}")
    
    # Test with string date
    string_date = "2025-05-10"
    features = generate_features(1, "PRODUCE", False, string_date)
    print(f"String date '{string_date}': Generated {len(features)} features")
    
    # Test with invalid date
    invalid_date = "invalid-date"
    features = generate_features(1, "PRODUCE", False, invalid_date)
    print(f"Invalid date '{invalid_date}': Generated {len(features)} features")
    
    # Check that the function doesn't crash with None
    none_date = None
    try:
        features = generate_features(1, "PRODUCE", False, none_date)
        print(f"None date: Generated {len(features)} features ✅")
    except Exception as e:
        print(f"❌ Error with None date: {str(e)}")

def test_promotion_feature():
    """Test promotion feature."""
    print("\n=== Testing Promotion Feature ===")
    date = datetime.now()
    
    # Test with promotion on
    features_promo_on = generate_features(1, "PRODUCE", True, date)
    if features_promo_on[0] == 1:
        print("✅ Promotion ON correctly set")
    else:
        print("❌ Promotion ON not set correctly")
    
    # Test with promotion off
    features_promo_off = generate_features(1, "PRODUCE", False, date)
    if features_promo_off[0] == 0:
        print("✅ Promotion OFF correctly set")
    else:
        print("❌ Promotion OFF not set correctly")

def test_edge_cases():
    """Test various edge cases to ensure no index errors."""
    print("\n=== Testing Edge Cases ===")
    date = datetime.now()
    
    # Test combinations of edge cases
    test_cases = [
        (54, "LADIESWEAR", True),   # Max store, last family, promo on
        (1, "AUTOMOTIVE", False),   # Min store, first family, promo off
        (0, "INVALID", None),       # Invalid store and family, None promo
        (9999, "VERY_LONG_FAMILY_NAME_THAT_DOESNT_EXIST", True),  # Extreme values
    ]
    
    for store_nbr, family, promo in test_cases:
        try:
            features = generate_features(store_nbr, family, promo, date)
            print(f"✅ Edge case store={store_nbr}, family={family}, promo={promo} - generated {len(features)} features without errors")
        except Exception as e:
            print(f"❌ Error with edge case store={store_nbr}, family={family}, promo={promo}: {str(e)}")

def run_all_tests():
    """Run all tests."""
    print("Starting feature generation tests...")
    
    test_store_features()
    test_family_features()
    test_date_features()
    test_promotion_feature()
    test_edge_cases()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    run_all_tests() 