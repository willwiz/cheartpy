import re


def get_var_index(prefix: str, names: list[str]) -> list[int]:
    p = re.compile(rf"{prefix}-(\d+).D")
    m = [p.match(s) for s in names]
    return sorted(set(int(w.group(1)) for w in m if w))
