from pathlib import Path

from cheartpy.cheart_parsing.pfile._regex import OutputPath, get_macro, get_output_path


def find_output_dir(pfile: Path) -> Path | None:
    text = pfile.read_text()
    macros = {
        m.name: m.value for line in text.splitlines() if (m := get_macro(line.strip())) is not None
    }
    path = None
    for line in text.splitlines():
        match get_output_path(line.strip()):
            case OutputPath(p):
                path = str(p)
                break
            case None:
                continue
    if not path:
        return None
    for k, v in macros.items():
        path = path.replace(f"#{k}", v)
    return Path(path) if path else None
