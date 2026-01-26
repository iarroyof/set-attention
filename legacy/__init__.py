import os
import warnings


warnings.warn(
    "legacy.mixed_ska_baseline is deprecated and will be removed. "
    "Use src.models.set_only for new set-only attention implementation.",
    DeprecationWarning,
    stacklevel=2,
)

if os.getenv("PRODUCTION_MODE", "0") == "1":
    raise ImportError(
        "Legacy mixed SKA implementation not available in production mode."
    )
