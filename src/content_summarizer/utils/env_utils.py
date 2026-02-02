import os
from collections.abc import Generator
from contextlib import contextmanager


@contextmanager
def suppress_locale_crash() -> Generator[None, None, None]:
    """Temporarily forces a safe locale to prevent PyAV/FFmpeg crashes.

    This context manager overrides the `LANG` and `LC_ALL` environment variables
    to 'C.UTF-8' for the duration of the context. This is required to prevent
    underlying C libraries (such as those used by PyAV) from crashing when
    decoding non-ASCII characters in log outputs on systems with specific
    locales (e.g., pt_BR).

    The original environment variables are safely restored upon exit.

    Yields:
        None: Control is yielded back to the caller with the modified environment.

    """
    old_lang = os.environ.get("LANG")
    old_lc = os.environ.get("LC_ALL")

    os.environ["LANG"] = "C.UTF-8"
    os.environ["LC_ALL"] = "C.UTF-8"

    try:
        yield
    finally:
        if old_lang:
            os.environ["LANG"] = old_lang
        else:
            os.environ.pop("LANG", None)

        if old_lc:
            os.environ["LC_ALL"] = old_lc
        else:
            os.environ.pop("LC_ALL", None)
