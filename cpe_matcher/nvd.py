"""NVD dictionary tools, with one controlled HTTP transport for nvdlib 0.9.

The pinned library's public search functions handle parameters and conversion.
Their private __get binding is replaced in a *local function copy*, not globally:
the upstream paginator can silently omit failed pages and its generator retries
403 forever. This adapter owns pagination, HTTP retries and completeness instead.
"""
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
import math
import logging
import os
import random
import threading
import time
from types import FunctionType
from urllib.parse import parse_qsl, quote_plus

import nvdlib
import requests

from .cpe import component, match_string, split_cpe
from .logging_setup import safe_text

logger = logging.getLogger(__name__)


class NvdServiceError(RuntimeError):
    def __init__(self, message: str):
        super().__init__('SERVICE_ERROR: NVD: ' + message)


@dataclass(frozen=True)
class NvdSettings:
    api_key: str = field(repr=False)
    min_interval: float = 1.0
    timeout: float = 30.0
    max_retries: int = 4
    max_pages: int = 100
    page_size: int = 1000

    def __post_init__(self):
        if not self.api_key:
            raise ValueError('Не задан NIST_API_KEY')
        if not math.isfinite(self.min_interval) or self.min_interval < .6:
            raise ValueError('NVD_MIN_INTERVAL должен быть >= 0.6')
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError('NVD_TIMEOUT должен быть положительным')
        if not 0 <= self.max_retries <= 20 or self.max_pages < 1 or not 1 <= self.page_size <= 2000:
            raise ValueError('Некорректные пределы NVD')

    @classmethod
    def from_env(cls):
        return cls(os.getenv('NIST_API_KEY', ''), float(os.getenv('NVD_MIN_INTERVAL', '1')),
                   float(os.getenv('NVD_TIMEOUT', '30')), int(os.getenv('NVD_MAX_RETRIES', '4')),
                   int(os.getenv('NVD_MAX_PAGES', '100')), int(os.getenv('NVD_PAGE_SIZE', '1000')))


class RateLimiter:
    """Even spacing plus a hard sliding window; every attempt is counted."""
    def __init__(self, interval, clock=time.monotonic, sleep=time.sleep):
        self.interval, self.clock, self.sleep = interval, clock, sleep
        self.starts = deque()

    def wait(self):
        now = self.clock()
        while self.starts and self.starts[0] <= now - 30:
            self.starts.popleft()
        target = max(now, self.starts[-1] + self.interval if self.starts else now)
        if len(self.starts) >= 50:
            target = max(target, self.starts[0] + 30)
        if target > now:
            self.sleep(target - now)
        self.starts.append(self.clock())


def cached(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        key = (method.__name__, args, tuple(sorted(kwargs.items())))
        with self.lock:
            if key not in self.cache:
                self.cache[key] = method(self, *args, **kwargs)
            return self.cache[key]
    return call


class NvdService:
    def __init__(self, settings: NvdSettings, request=None, clock=time.monotonic, sleep=time.sleep):
        self.settings = settings
        self.request = request or requests.get
        self.sleep = sleep
        self.limiter = RateLimiter(settings.min_interval, clock, sleep)
        self.lock = threading.RLock()
        self.cache = {}
        self.start = 0
        self.metadata = {}
        self.searchers = {}
        for kind, original in [('cpe', nvdlib.searchCPE), ('cve', nvdlib.searchCVE)]:
            namespace = dict(original.__globals__, __get=self._get)
            self.searchers[kind] = FunctionType(original.__code__, namespace,
                                                original.__name__, original.__defaults__, original.__closure__)

    @classmethod
    def from_env(cls):
        return cls(NvdSettings.from_env())

    def _http_reason(self, response):
        reason = f'HTTP {response.status_code}'
        message = response.headers.get('message', '')
        if message:
            # Never include response bodies, request headers, or URLs with credentials.
            message = safe_text(message.replace(self.settings.api_key, '[REDACTED]'))[:500]
            reason += f'; сообщение NVD: {message}'
        if response.status_code == 403:
            reason += '; доступ отклонён: возможна блокировка запросов или проблема ключа'
        return reason

    def _get(self, product, headers, parameters, limit=None, delay=None, proxies=None):
        # nvdlib pre-encodes selected parameters. Decode once and let requests
        # encode the mapping once; keywordSearch is pre-encoded by _page below.
        params = dict(parse_qsl('&'.join(f'{k}={v}' for k, v in parameters.items())))
        params['startIndex'] = self.start
        url = f'https://services.nvd.nist.gov/rest/json/{"cpes" if product == "cpe" else "cves"}/2.0'
        for attempt in range(self.settings.max_retries + 1):
            self.limiter.wait()
            logger.debug('NVD: HTTP-запрос %s, startIndex=%s, попытка %s/%s',
                         product, self.start, attempt + 1, self.settings.max_retries + 1)
            response = None
            try:
                response = self.request(url, params=params, headers=headers, timeout=self.settings.timeout)
                if response.status_code in (429, 500, 502, 503, 504):
                    raise requests.ConnectionError('temporary HTTP failure')
                if response.status_code in (401, 403):
                    raise NvdServiceError(self._http_reason(response))
                invalid_cpe = (response.status_code == 404
                               and 'invalid cpematchstring parameter' in
                               response.headers.get('message', '').casefold())
                if response.status_code == 400 or invalid_cpe:
                    raise ValueError('NVD отклонил параметры запроса: ' + self._http_reason(response))
                if not 200 <= response.status_code < 300:
                    raise NvdServiceError(self._http_reason(response))
                raw = response.json()
                path = 'products' if product == 'cpe' else 'vulnerabilities'
                if not isinstance(raw, dict) or not isinstance(raw.get(path), list):
                    raise NvdServiceError('некорректный ответ API')
                for field in ('startIndex', 'totalResults', 'resultsPerPage'):
                    if type(raw.get(field)) is not int or raw[field] < 0:
                        raise NvdServiceError('нет метаданных полноты ответа')
                self.metadata = raw
                return raw
            except (requests.ConnectionError, requests.Timeout) as exc:
                if response is not None:
                    reason = self._http_reason(response)
                elif isinstance(exc, requests.exceptions.SSLError):
                    reason = 'ошибка TLS/сертификата'
                elif isinstance(exc, requests.Timeout):
                    reason = 'таймаут запроса'
                else:
                    reason = 'ошибка соединения'
                if attempt >= self.settings.max_retries:
                    raise NvdServiceError(f'{reason}; исчерпаны попытки ({attempt + 1})') from None
                pause = min(60, 2 ** attempt) + random.uniform(0, .5)
                if response is not None:
                    retry = response.headers.get('Retry-After', '')
                    try:
                        seconds = float(retry)
                    except ValueError:
                        try:
                            seconds = (parsedate_to_datetime(retry) - datetime.now(timezone.utc)).total_seconds()
                        except (ValueError, TypeError, OverflowError):
                            seconds = 0
                    if math.isfinite(seconds):
                        pause = max(pause, seconds)
                logger.warning('NVD: %s; повтор через %.1f с',
                               reason, pause)
                self.sleep(pause)
            except requests.RequestException:
                raise NvdServiceError('ошибка HTTP или некорректный JSON') from None

    def _page(self, kind, start, limit, **filters):
        self.start = start
        if filters.get('keywordSearch'):
            filters['keywordSearch'] = quote_plus(filters['keywordSearch'])
        try:
            rows = self.searchers[kind](key=self.settings.api_key, delay=self.settings.min_interval,
                                       limit=limit, asDict=True, **filters)
        except (KeyError, TypeError) as exc:
            raise NvdServiceError('нарушена схема ответа API') from exc
        return rows, self.metadata

    def _pages(self, kind, **filters):
        start = 0
        for _ in range(self.settings.max_pages):
            rows, meta = self._page(kind, start, self.settings.page_size, **filters)
            if meta['startIndex'] != start or (not rows and start < meta['totalResults']):
                raise NvdServiceError('нарушена последовательность страниц')
            start += len(rows)
            complete = start >= meta['totalResults']
            yield rows, complete
            if complete:
                return

    @staticmethod
    def _evidence(row):
        return {'cpe': row['cpeName'], 'deprecated': row.get('deprecated'),
                'titles': [t['title'] for t in row.get('titles', [])][:2],
                'deprecatedBy': row.get('deprecatedBy', [])}

    @staticmethod
    def _triple(row):
        try:
            triple = split_cpe(row['cpeName'])[:3]
            if triple[0] not in ('a', 'o', 'h') or type(row.get('deprecated')) is not bool:
                raise ValueError('Invalid dictionary entry')
            return triple
        except (KeyError, TypeError, ValueError) as exc:
            raise NvdServiceError('некорректная запись словаря CPE') from exc

    @cached
    def search_cpe(self, keyword: str = '', part: str = '', vendor: str = '', product: str = '', limit: int = 50):
        if not 1 <= limit <= 200 or not any((keyword.strip(), part, vendor, product)):
            raise ValueError('Задайте критерии поиска и limit от 1 до 200')
        filters = {}
        if keyword.strip():
            filters['keywordSearch'] = keyword.strip()
        if any((part, vendor, product)):
            filters['cpeMatchString'] = match_string(part, vendor, product)
        groups = {}
        complete = False
        for rows, complete in self._pages('cpe', **filters):
            for row in rows:
                triple = self._triple(row)
                if triple not in groups and len(groups) >= limit:
                    return {'groups': list(groups.values()), 'incomplete': True}
                group = groups.setdefault(triple, dict(zip(('part', 'vendor', 'product'), triple), evidence=[]))
                evidence = self._evidence(row)
                # Preserve at least one active and one deprecated example.
                if len(group['evidence']) < 2 or not any(e['deprecated'] == evidence['deprecated'] for e in group['evidence']):
                    group['evidence'].append(evidence)
            if len(groups) >= limit and not complete:
                break
        return {'groups': list(groups.values()), 'incomplete': not complete}

    def _validate(self, part='', vendor='', product=''):
        filters = {'cpeMatchString': match_string(part, vendor, product)}
        evidence, replacements = [], []
        complete = False
        for rows, complete in self._pages('cpe', **filters):
            for row in rows:
                triple = self._triple(row)
                if any(expected and actual.casefold() != expected.casefold()
                       for actual, expected in zip(triple, (part, vendor, product))):
                    continue
                item = self._evidence(row)
                if row.get('deprecated') is False:
                    return dict(zip(('part', 'vendor', 'product'), triple), status='valid', evidence=[item])
                if row.get('deprecated') is not True:
                    raise NvdServiceError('в записи CPE отсутствует признак deprecated')
                if len(evidence) < 3:
                    evidence.append(item)
                for replacement in row.get('deprecatedBy', []):
                    if replacement not in replacements and len(replacements) < 20:
                        replacements.append(replacement)
        status = 'incomplete' if not complete else ('deprecated_only' if evidence else 'absent')
        return {'status': status, 'part': part, 'vendor': vendor, 'product': product,
                'evidence': evidence, 'replacements': replacements}

    @cached
    def validate_product(self, part: str, vendor: str, product: str):
        if part not in ('a', 'o', 'h'):
            raise ValueError('part должен быть a, o или h')
        component(vendor)
        component(product)
        return self._validate(part, vendor, product)

    @cached
    def validate_vendor(self, vendor: str):
        component(vendor)
        return self._validate(vendor=vendor)

    @cached
    def search_cve(self, keyword: str, limit: int = 10):
        if not keyword.strip() or not 1 <= limit <= 50:
            raise ValueError('Задайте keyword и limit от 1 до 50')
        rows, meta = self._page('cve', 0, limit, keywordSearch=keyword.strip())
        # Bound model context while preserving the logical tree and vulnerable
        # flags. A truncated configuration is explicitly marked as incomplete.
        remaining = 120
        truncated = False

        def compact(value):
            nonlocal remaining, truncated
            if remaining <= 0:
                truncated = True
                return None
            if isinstance(value, list):
                result = []
                for child in value:
                    item = compact(child)
                    if item is not None:
                        result.append(item)
                return result
            if isinstance(value, dict):
                remaining -= 1
                return {key: compact(item) if isinstance(item, (dict, list)) else item
                        for key, item in value.items()
                        if key in ('operator', 'negate', 'nodes', 'children', 'cpeMatch', 'criteria',
                                   'vulnerable', 'matchCriteriaId', 'versionStartIncluding',
                                   'versionStartExcluding', 'versionEndIncluding', 'versionEndExcluding')}
            return value

        items = []
        for row in rows:
            descriptions = row.get('descriptions', [])
            preferred = next((d for d in descriptions if d.get('lang') == 'en'), descriptions[0] if descriptions else {})
            items.append({'id': row['id'], 'description': preferred.get('value', '')[:1500],
                          'configurations': compact(row.get('configurations', []))})
        return {'items': items,
                'incomplete': len(rows) < meta['totalResults'] or truncated,
                'configurations_truncated': truncated,
                'notice': 'CPE-критерии CVE — гипотезы; vulnerable=false обозначает окружение. Нужна проверка словаря CPE.'}
