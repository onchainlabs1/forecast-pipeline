#!/usr/bin/env python
"""
Minimal encryption demo for evidence artifacts (AIMS_ENCRYPT_DEMO=1).

Creates a Fernet-encrypted ZIP of docs/evidence/* using a local key file.
Key is stored locally (not committed). This is portfolio-safe and for demo purposes only.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

try:
    from cryptography.fernet import Fernet
except Exception as _e:
    Fernet = None  # optional dependency

from src.aims.flags import aims_encrypt_demo_enabled


BASE = Path("docs/evidence")
OUT = Path("docs/evidence_encrypted.bin")
KEY_FILE = Path(".aims_demo_fernet.key")


def _load_or_create_key() -> bytes:
    if KEY_FILE.exists():
        return KEY_FILE.read_bytes()
    key = Fernet.generate_key()
    KEY_FILE.write_bytes(key)
    return key


def run() -> None:
    if not aims_encrypt_demo_enabled():
        print("AIMS_ENCRYPT_DEMO flag is OFF; skipping.")
        return
    if Fernet is None:
        print("cryptography not installed; run: pip install cryptography")
        return

    key = _load_or_create_key()
    f = Fernet(key)

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if BASE.exists():
            for p in BASE.rglob("*"):
                if p.is_file():
                    zf.write(p, arcname=p.relative_to(BASE))

    cipher = f.encrypt(mem.getvalue())
    OUT.write_bytes(cipher)
    print(f"Encrypted evidence written to {OUT} (key in {KEY_FILE}, do not commit).")


if __name__ == "__main__":
    run()


