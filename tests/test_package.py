from __future__ import annotations

import importlib.metadata

import brmisc as m


def test_version():
    assert importlib.metadata.version("brmisc") == m.__version__
