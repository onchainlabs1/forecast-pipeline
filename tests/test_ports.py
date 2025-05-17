#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test to validate port configuration across scripts.
"""

import os
import unittest
import sys
import importlib.util

class TestPortConfiguration(unittest.TestCase):
    """Test case for port configuration validation."""
    
    def test_api_port_from_env(self):
        """Test that the API port can be configured via environment variable."""
        os.environ["API_PORT"] = "9000"
        
        # Import run_with_landing as a module
        spec = importlib.util.spec_from_file_location(
            "run_with_landing", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_with_landing.py")
        )
        run_with_landing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_with_landing)
        
        # Check that the API_PORT was read from the environment
        self.assertEqual(run_with_landing.API_PORT, 9000)
        
        # Clean up
        del os.environ["API_PORT"]
    
    def test_dashboard_port_from_env(self):
        """Test that the dashboard port can be configured via environment variable."""
        os.environ["DASHBOARD_PORT"] = "9501"
        
        # Import run_with_landing as a module
        spec = importlib.util.spec_from_file_location(
            "run_with_landing", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_with_landing.py")
        )
        run_with_landing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_with_landing)
        
        # Check that the DASHBOARD_PORT was read from the environment
        self.assertEqual(run_with_landing.DASHBOARD_PORT, 9501)
        
        # Clean up
        del os.environ["DASHBOARD_PORT"]
    
    def test_landing_port_from_env(self):
        """Test that the landing port can be configured via environment variable."""
        os.environ["LANDING_PORT"] = "9002"
        
        # Import run_with_landing as a module
        spec = importlib.util.spec_from_file_location(
            "run_with_landing", 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "run_with_landing.py")
        )
        run_with_landing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(run_with_landing)
        
        # Check that the LANDING_PORT was read from the environment
        self.assertEqual(run_with_landing.LANDING_PORT, 9002)
        
        # Clean up
        del os.environ["LANDING_PORT"]

if __name__ == "__main__":
    unittest.main() 