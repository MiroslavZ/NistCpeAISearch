"""Console diagnostics, isolated from SDK logging and MCP stdout."""
import logging
import os
import sys


def safe_text(text):
    for name in ('LLM_API_KEY', 'NIST_API_KEY'):
        secret = os.getenv(name)
        if secret:
            text = text.replace(secret, '[REDACTED]')
    # External strings must not inject new log lines or terminal commands.
    return ''.join(char if char.isprintable() else f'\\x{ord(char):02x}' for char in text)


class ConsoleFormatter(logging.Formatter):
    def format(self, record):
        return safe_text(super().format(record))


def configure_logging(verbose=False):
    logger = logging.getLogger('cpe_matcher')
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ConsoleFormatter('%(asctime)s %(levelname)s %(message)s', '%H:%M:%S'))
    logger.addHandler(handler)
