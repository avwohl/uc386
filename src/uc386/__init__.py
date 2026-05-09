"""uc386 - C23 compiler for i386/MS-DOS."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("uc386")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
