__all__ = ["LOG_LEVEL", "LogLevel", "BLogger", "NullLogger", "ILogger", "bcolors"]
import abc
import enum
import os
from typing import Any, Literal, Mapping
from datetime import datetime
import traceback
from inspect import getframeinfo, stack


def now() -> str:
    return datetime.now().strftime("%H:%M:%S")


LOG_LEVEL = Literal["NULL", "FATAL", "ERROR", "WARN", "BRIEF", "INFO", "DEBUG"]


class LogLevel(enum.IntEnum):
    NULL = 0
    FATAL = 1
    ERROR = 2
    WARN = 3
    BRIEF = 4
    INFO = 5
    DEBUG = 6


class bcolors(enum.StrEnum):
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARN = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


LB: Mapping[LogLevel, str] = {
    LogLevel.NULL: "",
    LogLevel.FATAL: bcolors.FAIL,
    LogLevel.ERROR: bcolors.FAIL,
    LogLevel.WARN: bcolors.WARN,
    LogLevel.BRIEF: bcolors.OKCYAN,
    LogLevel.INFO: bcolors.OKGREEN,
    LogLevel.DEBUG: bcolors.OKBLUE,
}

RB: Mapping[LogLevel, str] = {
    LogLevel.NULL: "",
    LogLevel.FATAL: bcolors.ENDC,
    LogLevel.ERROR: bcolors.ENDC,
    LogLevel.WARN: bcolors.ENDC,
    LogLevel.BRIEF: bcolors.ENDC,
    LogLevel.INFO: bcolors.ENDC,
    LogLevel.DEBUG: bcolors.ENDC,
}


class ILogger(abc.ABC):
    @property
    @abc.abstractmethod
    def mode(self) -> LogLevel: ...
    @abc.abstractmethod
    def disp(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def debug(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def info(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def brief(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def warn(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def error(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def fatal(self, *msg: Any) -> None: ...
    @abc.abstractmethod
    def exception(self, e: Exception) -> Exception: ...


class BLogger(ILogger):
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

    def print(self, *msg: Any, level: LogLevel):
        if len(msg) < 1:
            return
        frame = getframeinfo(stack()[2][0])
        file = os.path.join(*frame.filename.split(os.sep)[-3:])
        print(
            f"\n[{now()}|{LB[level]}{level.name}{RB[level]}]({file}:{frame.lineno}|{frame.function})>>>"
        )
        for m in msg:
            print(m)

    def debug(self, *msg: Any) -> None:
        if self.level >= LogLevel.DEBUG:
            self.print(*msg, level=LogLevel.DEBUG)

    def info(self, *msg: Any) -> None:
        if self.level >= LogLevel.INFO:
            self.print(*msg, level=LogLevel.INFO)

    def disp(self, *msg: Any) -> None:
        if self.level >= LogLevel.INFO:
            print(*msg)

    def brief(self, *msg: Any) -> None:
        if self.level >= LogLevel.BRIEF:
            self.print(*msg, level=LogLevel.BRIEF)

    def warn(self, *msg: Any) -> None:
        if self.level >= LogLevel.WARN:
            self.print(*msg, level=LogLevel.WARN)

    def error(self, *msg: Any) -> None:
        print("Call Error")
        if self.level >= LogLevel.ERROR:
            self.print(*msg, level=LogLevel.ERROR)

    def fatal(self, *msg: Any) -> None:
        if self.level >= LogLevel.FATAL:
            self.print(*msg, level=LogLevel.FATAL)

    def exception(self, e: Exception):
        print(traceback.format_exc())
        return e


class NullLogger(ILogger):
    __slots__ = ["level"]
    level: LogLevel

    def __init__(self) -> None:
        self.level = LogLevel.NULL

    @property
    def mode(self) -> LogLevel:
        return self.level

    def print(self, *msg: Any, level: LogLevel) -> None:
        pass

    def debug(self, *msg: Any) -> None:
        pass

    def info(self, *msg: Any) -> None:
        pass

    def disp(self, *msg: Any) -> None:
        pass

    def brief(self, *msg: Any) -> None:
        pass

    def warn(self, *msg: Any) -> None:
        pass

    def error(self, *msg: Any) -> None:
        pass

    def fatal(self, *msg: Any) -> None:
        pass

    def exception(self, e: Exception) -> Exception:
        return e
