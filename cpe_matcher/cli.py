import argparse
import asyncio
from collections import Counter
import hashlib
import logging
import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .agent import Matcher, STATUSES
from .config import Settings
from .llm import DeepSeek, ServiceError
from .logging_setup import configure_logging
from .nvd import NvdSettings
from .storage import RunLock, State, export_report, read_report

logger = logging.getLogger(__name__)


async def process(rows, state, settings, verbose):
    pending = [(index, row) for index, row in enumerate(rows, 1) if state.get(row) is None]
    logger.info('Строк: %s; сохранено ранее: %s; осталось: %s',
                len(rows), len(rows) - len(pending), len(pending))
    if not pending:
        return
    llm = DeepSeek(settings)
    parameters = StdioServerParameters(command=sys.executable,
        args=['-m', 'cpe_matcher.mcp_server'],
        env=dict(os.environ, CPE_MATCH_VERBOSE='1' if verbose else '0'))
    try:
        logger.info('Запуск локального MCP-сервера и подключение')
        async with stdio_client(parameters) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                logger.info('MCP-сервер подключён; получение инструментов')
                tools = (await session.list_tools()).tools
                required = {'search_cpe', 'search_cve', 'validate_product', 'validate_vendor'}
                if not required.issubset({tool.name for tool in tools}):
                    raise RuntimeError('MCP-сервер не предоставляет обязательные инструменты')
                matcher = Matcher(llm, session, tools, settings)
                for index, row in pending:
                    logger.info('[%s/%s] ПО: %s | производитель: %s | версия: %s',
                                index, len(rows), row.input_product, row.input_vendor, row.input_version)
                    if state.get(row) is None:
                        state.save(row, await matcher.match(row))
                    else:
                        logger.info('Используется сохранённый результат идентичной строки')
                    logger.info('[%s/%s] Результат сохранён: %s',
                                index, len(rows), state.get(row)['status'])
    finally:
        await llm.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Сопоставление ПО с продуктами CPE NVD')
    parser.add_argument('input', type=Path, help='Исходный CSV UTF-8')
    parser.add_argument('-o', '--output', type=Path, required=True, help='Выходной CSV')
    parser.add_argument('--state', type=Path, help='SQLite-состояние (по умолчанию OUTPUT.sqlite3)')
    parser.add_argument('--resume', action='store_true', help='Продолжить сохранённую обработку')
    parser.add_argument('--retry-status', action='append', choices=STATUSES, default=[],
                        help='Повторить статус при --resume (можно указать несколько раз)')
    parser.add_argument('--env-file', type=Path, default=Path('.env'))
    parser.add_argument('--verbose', action='store_true', help='Подробные логи: аргументы инструментов и страницы NVD')
    args = parser.parse_args(argv)
    try:
        load_dotenv(args.env_file, override=False)
        configure_logging(args.verbose)
        settings = Settings.from_env()
        NvdSettings.from_env()
        state_path = args.state or Path(str(args.output) + '.sqlite3')
        paths = [p.resolve() for p in (args.input, args.output, state_path, args.env_file)]
        if len(set(paths)) != len(paths):
            raise ValueError('Пути входа, выхода, состояния и конфигурации должны различаться')
        if args.retry_status and not args.resume:
            raise ValueError('--retry-status требует --resume')
        logger.info('Чтение входного CSV: %s', args.input)
        rows = read_report(args.input)
        digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
        with RunLock([settings.api_key, settings.nist_key]):
            with State(state_path, digest, settings.fingerprint(), resume=args.resume) as state:
                if args.resume:
                    logger.info('Продолжение обработки из сохранённого состояния')
                    state.reset_statuses({'search_incomplete', *args.retry_status})
                try:
                    asyncio.run(process(rows, state, settings, args.verbose))
                finally:
                    logger.info('Сохранение выходного CSV: %s', args.output)
                    export_report(args.output, rows, state)
                    counts = Counter((state.get(row) or {}).get('status', 'search_incomplete') for row in rows)
                    logger.info('Итоги: %s', ', '.join(f'{key}={count}' for key, count in sorted(counts.items())))
        return 0
    except KeyboardInterrupt:
        print('Обработка прервана. Для продолжения используйте --resume.', file=sys.stderr)
        return 130
    except Exception as exc:
        # Provider exceptions may contain credentials or full request bodies.
        while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
            exc = exc.exceptions[0]
        if isinstance(exc, (ValueError, FileNotFoundError, ServiceError)):
            message = str(exc)
        else:
            message = 'Сбой сервиса или файловой операции. Проверьте настройки и повторите с --resume.'
        for name in ('LLM_API_KEY', 'NIST_API_KEY'):
            secret = os.getenv(name)
            if secret:
                message = message.replace(secret, '[REDACTED]')
        print('Ошибка: ' + message, file=sys.stderr)
        return 1
