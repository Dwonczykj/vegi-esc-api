from enum import Enum
from typing import Callable, Literal, Any
from pprint import pprint
from colorama import Fore, Style
try:
    from IPython.core.display import HTML as html_print
    from icecream import ic, IceCreamDebugger
    only_pprint = True
except:
    only_pprint = False
    ic = pprint
    html_print = pprint
    IceCreamDebugger = Callable[[str],None]
    
# Initialize logging.
import logging
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


global LOG_LEVEL

LOG_LEVEL: int = 3

class LogLevel(Enum):
    off = 0
    error = 1
    warn = 2
    info = 3
    verbose = 4
    
def set_log_level(level:LogLevel):
    LOG_LEVEL = level.value
    return LOG_LEVEL


PrintTypeLiteral = Literal['sys', 'ic', 'display', 'print', 'html_print']


def log_return_with(val: Any, log_level: Literal[0, 1, 2, 3, 4] = LogLevel.verbose.value, string_color: str = Fore.BLACK, printer: PrintTypeLiteral = 'display'):
    if log_level <= LogLevel.error.value:
        string_color = Fore.RED
    elif log_level <= LogLevel.warn.value:
        string_color = Fore.YELLOW
    elif log_level <= LogLevel.info.value:
        string_color = Fore.BLUE
        
    c: Any = print
    if printer == 'display':
        c = pprint
    elif printer == 'sys':
        c = pprint
    elif printer == 'print':
        c = print
    elif only_pprint:
        if printer == 'ic':
            c = ic
        elif printer == 'html_print':
            c = html_print

    if log_level <= LOG_LEVEL:
        try:
            if hasattr(val, 'to_html') and string_color != Fore.BLACK and printer in ['display', 'html_print']:
                html_print(
                    f'<span style="color:{string_color.lower()}">{val.to_html()}</span>')
            else:
                c(string_color + val + Style.RESET_ALL)
        except:
            c(val)
    return val


def log_return(val: Any):
    return log_return_with(val)


def log_with(val: Any, log_level: Literal[0, 1, 2, 3, 4] = LogLevel.verbose.value, string_color: str = Fore.BLACK, printer: PrintTypeLiteral = 'display'):
    log_return_with(val, log_level, string_color, printer)


def error(val: Any):
    return log_with(val, log_level=LogLevel.error.value)


def warn(val: Any):
    return log_with(val, log_level=LogLevel.warn.value)


def info(val: Any):
    return log_with(val, log_level=LogLevel.info.value)


def verbose(val: Any):
    return log_with(val, log_level=LogLevel.verbose.value)


def highlight_substring(s: str, substring_start: int, substring_end: int, highlight_color: str = Fore.LIGHTYELLOW_EX, string_color: str = Fore.BLACK, printer: PrintTypeLiteral = 'ic', out_string: bool = False):
    if only_pprint:
        # c: Callable[[str], None] = pprint
        c = pprint
    else:
        c = ic
    if printer == 'display':
        c = pprint
    elif printer == 'sys':
        c = pprint
    elif printer == 'html_print':
        c = html_print # type: ignore
    ss = string_color + s[:substring_start] + highlight_color + s[substring_start:substring_end] + (
        highlight_color if highlight_color == '**' else string_color) + s[substring_end:]
    if out_string:
        return ss
    return c(ss)


def highlight_substring_jnb(s: str, substring_start: int, substring_end: int, printer: PrintTypeLiteral = 'ic'):
    return highlight_substring(s, substring_start, substring_end, highlight_color='**', string_color='', printer='display', out_string=True)


highlight_substring_jnb('Testing Highlighter in strings', 8, 19, printer='sys')
