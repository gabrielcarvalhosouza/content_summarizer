import locale
import os
from collections.abc import Generator
from contextlib import contextmanager, suppress


@contextmanager
def suppress_locale_crash() -> Generator[None, None, None]:
    """Temporarily forces a safe locale to prevent PyAV/FFmpeg crashes.

    This context manager overrides both the process-wide locale (via `setlocale`)
    and the environment variables to 'C' or 'C.UTF-8'. This is critical for
    libraries like PyAV that link directly to C runtimes, as they read the
    active locale configuration, not just environment variables.

    The original locale settings are strictly restored upon exit.

    Yields:
        None: Control is yielded back to the caller with the modified environment.

    """
    old_env_lang = os.environ.get("LANG")
    old_env_lc = os.environ.get("LC_ALL")

    old_locale = None
    with suppress(locale.Error):
        old_locale = locale.setlocale(locale.LC_ALL, None)

    os.environ["LANG"] = "C"
    os.environ["LC_ALL"] = "C"

    try:
        locale.setlocale(locale.LC_ALL, "C")
    except locale.Error:
        with suppress(locale.Error):
            locale.setlocale(locale.LC_ALL, "C.UTF-8")

    try:
        yield
    finally:
        if old_env_lang is not None:
            os.environ["LANG"] = old_env_lang

        if old_env_lang is None:
            os.environ.pop("LANG", None)

        if old_env_lc is not None:
            os.environ["LC_ALL"] = old_env_lc

        if old_env_lc is None:
            os.environ.pop("LC_ALL", None)

        if old_locale:
            with suppress(locale.Error):
                locale.setlocale(locale.LC_ALL, old_locale)
