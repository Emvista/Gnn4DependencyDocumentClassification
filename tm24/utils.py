"""Miscelaneous utilities."""

import importlib

def dyn_load(name, item=None):
    return getattr(
        importlib.import_module(name[:name.rindex(".")]),
        name[name.rindex(".")+1:] if item is None else item
    )
