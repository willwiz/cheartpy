__all__ = ["LogLevel", "BasicLogger", "NullLogger", "_Logger"]
import abc
import enum
from typing import Any, Literal
from datetime import datetime
import traceback
from inspect import getframeinfo, stack


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


class LogLevel(enum.IntEnum):
    NULL = 0
    FATAL = 1
    ERROR = 2
    WARN = 3
    BRIEF = 4
    INFO = 5
    DEBUG = 6


class _Logger(abc.ABC):
    @property
    @abc.abstractmethod
    def mode(self) -> LogLevel: ...
    @abc.abstractmethod
    def debug(self, msg: Any): ...
    @abc.abstractmethod
    def info(self, msg: Any): ...
    @abc.abstractmethod
    def brief(self, msg: Any): ...
    @abc.abstractmethod
    def warn(self, msg: Any): ...
    @abc.abstractmethod
    def error(self, msg: Any): ...
    @abc.abstractmethod
    def fatal(self, msg: Any): ...
    @abc.abstractmethod
    def exception(self, e: Exception) -> Exception: ...


class BasicLogger(_Logger):
    __slots__ = ["level"]
    level: LogLevel

    def __init__(
        self,
        level: (
            LogLevel
            | Literal["NULL", "FATAL", "ERROR", "WARN", "BRIEF", "INFO", "DEBUG"]
        ),
    ) -> None:
        self.level = level if isinstance(level, LogLevel) else LogLevel[level]

    @property
    def mode(self) -> LogLevel:
        return self.level

    def print(self, msg: Any, level: LogLevel):
        frame = getframeinfo(stack()[2][0])
        print(f"[{now()}][{level.name:5}]::{frame.function}-{frame.lineno}>>>\n{msg}\n")

    def debug(self, msg: Any):
        if self.level >= LogLevel.DEBUG:
            self.print(msg, LogLevel.DEBUG)

    def info(self, msg: Any):
        if self.level >= LogLevel.INFO:
            self.print(msg, LogLevel.INFO)

    def brief(self, msg: Any):
        if self.level >= LogLevel.BRIEF:
            self.print(msg, LogLevel.BRIEF)

    def warn(self, msg: Any):
        if self.level >= LogLevel.WARN:
            self.print(msg, LogLevel.WARN)

    def error(self, msg: Any):
        if self.level >= LogLevel.ERROR:
            self.print(msg, LogLevel.ERROR)

    def fatal(self, msg: Any):
        if self.level >= LogLevel.FATAL:
            self.print(msg, LogLevel.FATAL)

    def exception(self, e: Exception):
        print(traceback.format_exc())
        return e


class NullLogger(_Logger):
    __slots__ = ["level"]
    level: LogLevel

    def __init__(self, level: LogLevel = LogLevel.NULL) -> None:
        self.level = LogLevel.NULL

    @property
    def mode(self) -> LogLevel:
        return self.level

    def print(self, msg: Any, level: LogLevel):
        pass

    def debug(self, msg: Any):
        pass

    def info(self, msg: Any):
        pass

    def brief(self, msg: Any):
        pass

    def warn(self, msg: Any):
        pass

    def error(self, msg: Any):
        pass

    def fatal(self, msg: Any):
        pass

    def exception(self, e: Exception):
        return e
