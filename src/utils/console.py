"""Cross-platform UTF-8 console configuration.

Windows PowerShell 5.1 commonly starts with OEM code page 437 while redirected
Python output defaults to a legacy ANSI encoding.  The mismatch corrupts
Vietnamese text even when the source files themselves are valid UTF-8.
"""

import os
import sys


def configure_utf8_console() -> None:
    """Use UTF-8 for the Windows console and Python standard streams."""
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleCP(65001)
            kernel32.SetConsoleOutputCP(65001)
        except (AttributeError, OSError):
            # Stream reconfiguration below is still sufficient for redirected
            # output and modern terminals where no Win32 console is attached.
            pass

    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            kwargs = {"encoding": "utf-8", "errors": "replace"}
            if stream is not sys.stdin:
                kwargs["line_buffering"] = True
            reconfigure(**kwargs)
