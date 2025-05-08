#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to initialize the database and load initial data.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parents[2]
sys.path.insert(0, str(project_root))

from src.database.database import init_db, db_session
from src.database.data_loader import load_initial_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    """Initialize database and load initial data."""
    try:
        logger.info("Initializing database")
        
        # Initialize database (create tables)
        init_db()
        logger.info("Database initialized successfully")
        
        # Load initial data
        with db_session() as session:
            load_initial_data(session)
        
        logger.info("Database setup completed successfully")
    
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 