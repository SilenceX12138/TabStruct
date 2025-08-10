from datetime import datetime


class TerminalIO:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    ENDC = "\033[0m"

    @classmethod
    def print(cls, message, color, highlight=False):
        prefix = color + cls.BOLD + f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]" + " Codebase: " + cls.ENDC
        # Print colored + bold prefix
        print(prefix, end="")
        # Print highlighted message if required
        message = ">" * 10 + message + "<" * 10 if highlight else message
        # Print colored message
        print(f"{color}{message}{cls.ENDC}")

    @classmethod
    def trace_func(cls, func):
        def wrapper(*args, **kwargs):
            TerminalIO.print(f"Lauching {func.__name__}()", color=TerminalIO.OKGREEN, highlight=True)
            result = func(*args, **kwargs)
            TerminalIO.print(f"Exiting from {func.__name__}()", color=TerminalIO.OKGREEN, highlight=True)
            return result

        return wrapper
