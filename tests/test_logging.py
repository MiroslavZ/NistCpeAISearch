import contextlib
import io
import logging
import unittest
from unittest.mock import patch

from cpe_matcher.logging_setup import configure_logging


class LoggingTests(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger('cpe_matcher')
        self.original = (self.logger.handlers[:], self.logger.level, self.logger.propagate)

    def tearDown(self):
        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers, level, self.logger.propagate = self.original
        self.logger.setLevel(level)

    def test_stderr_redacts_secrets_and_neutralizes_inventory_control_characters(self):
        stderr, stdout = io.StringIO(), io.StringIO()
        with patch.dict('os.environ', {'LLM_API_KEY': 'secret-one', 'NIST_API_KEY': 'secret-two'}):
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(stdout):
                configure_logging()
                logging.getLogger('cpe_matcher.agent').info('ПО: secret-one\nsecret-two\x1b[31m')
        self.assertEqual(stdout.getvalue(), '')
        self.assertNotIn('secret-', stderr.getvalue())
        self.assertEqual(stderr.getvalue().count('[REDACTED]'), 2)
        self.assertEqual(len(stderr.getvalue().splitlines()), 1)
        self.assertIn(r'\x1b', stderr.getvalue())

    def test_verbose_controls_details_and_reconfiguration_does_not_duplicate(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            configure_logging()
            self.logger.debug('hidden-details')
            self.logger.info('visible-state')
            configure_logging(True)
            self.logger.debug('shown-details')
        self.assertNotIn('hidden-details', stderr.getvalue())
        self.assertEqual(stderr.getvalue().count('visible-state'), 1)
        self.assertEqual(stderr.getvalue().count('shown-details'), 1)
