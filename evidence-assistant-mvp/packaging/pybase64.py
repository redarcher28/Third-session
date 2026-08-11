"""Pure-Python compatibility shim for the ``pybase64`` package.

Provides the small subset of the pybase64 API used by chromadb, implemented
with the standard library :mod:`base64` module.  Install only on Python 3.13
where upstream pybase64 has no compatible release:

    Copy-Item packaging\\pybase64.py .venv\\Lib\\site-packages\\pybase64.py
"""

from __future__ import annotations

import base64 as _base64

__version__ = "1.4.1"


def b64encode(data, altchars=None):
    """Encode bytes and return bytes (mirrors pybase64.b64encode)."""
    return _base64.b64encode(data, altchars=altchars)


def b64encode_as_string(data, altchars=None):
    """Encode bytes and return an ASCII ``str`` (mirrors pybase64.b64encode_as_string)."""
    return _base64.b64encode(data, altchars=altchars).decode("ascii")


def b64decode(data, altchars=None, validate=False):
    """Decode base64 data and return bytes (mirrors pybase64.b64decode)."""
    return _base64.b64decode(data, altchars=altchars, validate=validate)


def b64decode_as_bytes(data, altchars=None, validate=False):
    """Decode base64 data and return bytes."""
    return _base64.b64decode(data, altchars=altchars, validate=validate)
