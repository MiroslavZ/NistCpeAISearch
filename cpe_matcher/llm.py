"""Sequential DeepSeek transport; SDK retries are explicitly disabled."""
import asyncio
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import time

from openai import APIConnectionError, APIStatusError, AsyncOpenAI

logger = logging.getLogger(__name__)


class ServiceError(RuntimeError):
    """Stop the entire run, retaining the checkpoint."""


class IncompleteResponseError(RuntimeError):
    """A provider stopped generation; asking it to fix an empty JSON is misleading."""


def retry_after(value):
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            return max(0.0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


class DeepSeek:
    def __init__(self, settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url,
                                  timeout=settings.timeout, max_retries=0)
        self.next_request = 0.0
        self.lock = asyncio.Lock()

    async def close(self):
        await self.client.close()

    async def complete(self, messages, tools):
        async with self.lock:
            for attempt in range(self.settings.retries + 1):
                await asyncio.sleep(max(0.0, self.next_request - time.monotonic()))
                self.next_request = time.monotonic() + self.settings.llm_interval
                try:
                    logger.info('DeepSeek: запрос отправляется, ожидание ответа (попытка %s/%s)',
                                attempt + 1, self.settings.retries + 1)
                    response = await self.client.chat.completions.create(
                        model=self.settings.model, messages=messages, tools=tools)
                    logger.info('DeepSeek: ответ получен')
                    if not response.choices:
                        raise ServiceError('DeepSeek вернул пустой ответ')
                    choice = response.choices[0]
                    logger.debug('DeepSeek: finish_reason=%s', choice.finish_reason)
                    if choice.finish_reason == 'insufficient_system_resource':
                        raise ServiceError('DeepSeek прервал генерацию: insufficient_system_resource')
                    if choice.finish_reason in ('length', 'content_filter', 'aborted'):
                        raise IncompleteResponseError(
                            f'DeepSeek не завершил ответ: finish_reason={choice.finish_reason}.')
                    message = choice.message.model_dump(exclude_none=True)
                    return {k: v for k, v in message.items()
                            if k in ('role', 'content', 'tool_calls', 'reasoning_content')}
                except (APIConnectionError, APIStatusError) as exc:
                    status = getattr(exc, 'status_code', None)
                    if status is not None and status not in (408, 409, 429) and status < 500:
                        raise ServiceError(f'DeepSeek: ошибка доступа или запроса HTTP {status}') from None
                    if attempt == self.settings.retries:
                        raise ServiceError('DeepSeek недоступен после повторных попыток') from None
                    headers = getattr(getattr(exc, 'response', None), 'headers', {})
                    delay = max(retry_after(headers.get('retry-after')), 2 ** (attempt + 1) + random.random())
                    logger.warning('DeepSeek: %s; повтор через %.1f с',
                                   f'HTTP {status}' if status else 'ошибка соединения', delay)
                    self.next_request = max(self.next_request, time.monotonic() + delay)
