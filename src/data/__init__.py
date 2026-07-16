__all__ = ["Wikitext2Dataset"]


def __getattr__(name):
    if name == "Wikitext2Dataset":
        from .wikitext2 import Wikitext2Dataset

        return Wikitext2Dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
