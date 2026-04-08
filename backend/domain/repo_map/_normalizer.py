"""Symbol deduplication."""

from ._loaders import _Symbol


def _dedupe_symbols(symbols: list[_Symbol]) -> list[_Symbol]:
    seen: set[tuple[str, str, int, int, str | None]] = set()
    out: list[_Symbol] = []
    for name, kind, line_start, line_end, signature, parent, source in symbols:
        key = (name, kind.value, line_start, line_end, parent)
        if key in seen:
            continue
        seen.add(key)
        out.append((name, kind, line_start, line_end, signature, parent, source))
    return out
