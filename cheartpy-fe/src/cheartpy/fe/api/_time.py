from pathlib import Path

from cheartpy.fe.impl import TimeScheme


def create_time_scheme(
    name: str,
    start: int,
    stop: int,
    step: float | str | Path,
) -> TimeScheme:
    if isinstance(step, str) and not Path(step).is_file():
        msg = f"Time step file {step} is not found!"
        raise ValueError(msg)
    return TimeScheme(name, start, stop, step)
