#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database connection and session management.
"""

import os
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

from src.database.models import Base

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Define paths
PROJECT_DIR = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_DIR / "data" / "db"

# Create directory if it doesn't exist
os.makedirs(DB_DIR, exist_ok=True)

# Database URL
DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_DIR}/sales_forecasting.db")

# SQLAlchemy engine
if DB_URL.startswith("sqlite"):
    # For SQLite, set check_same_thread to False to allow multiple threads
    engine = create_engine(
        DB_URL, 
        connect_args={"check_same_thread": False},
        echo=False,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20
    )
else:
    # For other databases (PostgreSQL, MySQL, etc.)
    engine = create_engine(
        DB_URL, 
        echo=False,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_recycle=3600  # Recycle connections after 1 hour
    )

# Session maker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get a database session.
    
    Yields
    ------
    Session
        SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """
    Context manager for database sessions.
    
    Yields
    ------
    Session
        SQLAlchemy session.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        session.close()


def init_db() -> None:
    """
    Initialize the database by creating all tables.
    """
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def drop_db() -> None:
    """
    Drop all database tables.
    Only use this in development/testing!
    """
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("All database tables dropped")
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise 