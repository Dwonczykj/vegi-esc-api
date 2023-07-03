from enum import Enum
from typing import Callable, Literal, Any, Type
import os
import sys
from types import TracebackType
import traceback
from pprint import pprint, pformat
from colorama import Fore, Style
import time


try:
    from IPython.core.display import HTML as html_print
    from icecream import ic  # , IceCreamDebugger
    only_pprint = True
except Exception:
    only_pprint = False
    ic = pprint
    html_print = pprint
    IceCreamDebugger = Callable[[str], None]

# Initialize logging.
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


global LOG_LEVEL

LOG_LEVEL: int = 5


class LogLevel(Enum):
    off = 0
    error = 1
    warn = 2
    log = 3
    info = 4
    verbose = 5


def set_log_level(level: LogLevel):
    LOG_LEVEL = level.value
    return LOG_LEVEL


PrintTypeLiteral = Literal["sys", "ic", "display", "print", "html_print"]


def timeis(func: Callable):
    '''Decorator that reports the execution time.'''

    def wrap(*args: Any, **kwargs: Any):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print(func.__name__, end - start)
        return result
    return wrap


def _slow_call_timer(func: Callable, n: int = 5):
    '''Decorator to log warn function calls that take longer than 5 seconds'''
    def wrap(*args: Any, **kwargs: Any):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        if (end - start) > n:
            warn(f'{func.__name__} took {end-start} seconds to run.')
        return result
    return wrap


def slow_call_timer_n(n: int):
    '''Decorator to log warn function calls that take longer than N seconds'''
    def wrap(func: Callable):
        return _slow_call_timer(func=func, n=n)
    return wrap


def slow_call_timer(func: Callable):
    '''Decorator to log warn function calls that take longer than 5 seconds'''
    return _slow_call_timer(func=func, n=5)


__mycode = True


def is_mycode(tb: TracebackType):
    globals = tb.tb_frame.f_globals
    return '__mycode' in globals


def mycode_traceback_levels(tb: TracebackType | None):
    '''
    Extract your frames from your codebase:
        
        1. "skip the frames that don't matter to you (e.g. custom assert code)"
        
        2. "identify how many frames are part of your code -> length"
        
        3. "extract length frames"
    '''
    length = 0
    while tb and is_mycode(tb):
        tb = tb.tb_next
        length += 1
    return length


def format_exception(type: Type[BaseException] | None, value: BaseException | None, tb: TracebackType | None):
    # 1. skip custom assert code, e.g.
    # while tb and is_custom_assert_code(tb):
    #   tb = tb.tb_next
    # 2. only display your code
    length = mycode_traceback_levels(tb)
    return ''.join(traceback.format_exception(type, value, tb, length))


def format_full_stacktrace(log_level: Literal[0, 1, 2, 3, 4, 5]):
    '''
    ~ https://stackoverflow.com/a/32999522
    '''
    # print(__file__)
    # print(os.path.abspath(__file__))
    # print(os.path.realpath(__file__))
    # print(os.path.relpath(os.path.dirname(__file__)))
    # # print(list(__path__))
    # print(__package__)
    # print(os.path.abspath(__package__))
    # print(os.path.realpath(__package__))
    # print(__loader__)
    src_path_descriptor = os.path.relpath(os.path.dirname(__file__))
    stackTraceFrames = [SF for SF in traceback.extract_stack(f=None, limit=None) if src_path_descriptor in SF.filename and '/Users/joey/.vscode/extensions' not in SF.filename and not SF.filename.endswith('logger.py')]
    if log_level <= LogLevel.warn.value:
        stackTraceStrList = pformat(traceback.format_list(stackTraceFrames))
    else:
        stackTraceStrList = pformat(traceback.format_list(stackTraceFrames[:-2]))
    return stackTraceStrList


def log_return_with(
    val: Any,
    log_level: Literal[0, 1, 2, 3, 4, 5] = LogLevel.verbose.value,
    string_color: str = Fore.BLACK,
    printer: PrintTypeLiteral = "display",
    stackTrace: list[str] | None = None,
):
    if log_level <= LogLevel.error.value:
        string_color = Fore.RED
    elif log_level <= LogLevel.warn.value:
        string_color = Fore.YELLOW
    elif log_level <= LogLevel.info.value:
        string_color = Fore.BLUE

    c: Any = print
    if printer == "display":
        def _x(v: Any):
            return pprint(v, indent=2)
        c = _x
    elif printer == "sys":
        def _x(v: Any):
            return pprint(v, indent=2)
        c = pprint
    elif printer == "print":
        c = print
    elif only_pprint:
        if printer == "ic":
            c = ic
        elif printer == "html_print":
            c = html_print

    if not stackTrace:
        # Each item in the list is a quadruple (filename, line number, function name, text), and the entries are in order from oldest to newest stack frame.
        # stackTraceList = [text for (filename, line_number, function_name, text) in traceback.extract_stack(f=None, limit=None) if '/Users/joey/.vscode/extensions' not in filename]
        # stackTrace = str(traceback.extract_stack(f=None, limit=None))
        # stackTrace = str(pformat('\n\t'.join(format_full_stacktrace(log_level=log_level))))
        stackTrace = format_full_stacktrace(log_level=log_level)

    if log_level <= LOG_LEVEL:
        try:
            if (
                hasattr(val, "to_html")
                and string_color != Fore.BLACK
                and printer in ["display", "html_print"]
            ):
                html_print(
                    f'<span style="color:{string_color.lower()}">{val.to_html()}</span>'
                )
            elif isinstance(val, str):
                c(string_color + val + Style.RESET_ALL)
                c(Fore.YELLOW + 'StackTrace: ' + Style.RESET_ALL)
                print(stackTrace)
            else:
                c(string_color + str(val) + Style.RESET_ALL)
                c(Fore.YELLOW + 'StackTrace: ' + Style.RESET_ALL)
                # c(Fore.YELLOW + stackTrace + Style.RESET_ALL)
                print(stackTrace)
        except Exception as e:
            print(Fore.YELLOW + 'StackTrace: ' + Style.RESET_ALL)
            print(stackTrace)
            print(e)
            c(val + Style.RESET_ALL)
    return val


def log_return(val: Any):
    return log_return_with(val)


def log_with(
    val: Any,
    log_level: Literal[0, 1, 2, 3, 4, 5] = LogLevel.verbose.value,
    string_color: str = Fore.BLACK,
    printer: PrintTypeLiteral = "display",
):
    log_return_with(val, log_level, string_color, printer)


def error(val: Any):
    print(val)
    exc = sys.exception()
    print("*** print_tb:")
    traceback.print_tb(exc.__traceback__, limit=1, file=sys.stdout)
    print("*** print_exception:")
    traceback.print_exception(exc, limit=2, file=sys.stdout)
    print("*** print_exc:")
    traceback.print_exc(limit=2, file=sys.stdout)
    print("*** format_exc, first and last line:")
    formatted_lines = traceback.format_exc().splitlines()
    print(formatted_lines[0])
    print(formatted_lines[-1])
    print("*** format_exception:")
    print(repr(traceback.format_exception(exc)))
    print("*** extract_tb:")
    print(repr(traceback.extract_tb(exc.__traceback__)))
    print("*** format_tb:")
    print(repr(traceback.format_tb(exc.__traceback__)))
    print("*** tb_lineno:", exc.__traceback__.tb_lineno)
    return log_with(val, log_level=LogLevel.error.value)


def warn(val: Any):
    return log_with(val, log_level=LogLevel.warn.value)


def log(val: Any):
    return log_with(val, log_level=LogLevel.log.value)


def info(val: Any):
    return log_with(val, log_level=LogLevel.info.value)


def verbose(val: Any):
    return log_with(val, log_level=LogLevel.verbose.value)


def highlight_substring(
    s: str,
    substring_start: int,
    substring_end: int,
    highlight_color: str = Fore.LIGHTYELLOW_EX,
    string_color: str = Fore.BLACK,
    printer: PrintTypeLiteral = "ic",
    out_string: bool = False,
):
    if only_pprint:
        # c: Callable[[str], None] = pprint
        c = pprint
    else:
        c = ic
    if printer == "display":
        c = pprint
    elif printer == "sys":
        c = pprint
    elif printer == "html_print":
        c = html_print  # type: ignore
    ss = (
        string_color
        + s[:substring_start]
        + highlight_color
        + s[substring_start:substring_end]
        + (highlight_color if highlight_color == "**" else string_color)
        + s[substring_end:]
    )
    if out_string:
        return ss
    return c(ss)


def highlight_substring_jnb(
    s: str, substring_start: int, substring_end: int, printer: PrintTypeLiteral = "ic"
):
    return highlight_substring(
        s,
        substring_start,
        substring_end,
        highlight_color="**",
        string_color="",
        printer="display",
        out_string=True,
    )


highlight_substring_jnb("Testing Highlighter in strings", 8, 19, printer="sys")
