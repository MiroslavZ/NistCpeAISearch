from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    base_url: str
    model: str
    nist_key: str = field(repr=False)
    max_steps: int = 40
    max_tools: int = 80
    llm_interval: float = 1.0
    timeout: float = 120.0
    retries: int = 4
    mcp_timeout: float = 3600.0

    @classmethod
    def from_env(cls):
        names = ['LLM_API_KEY', 'LLM_BASE_URL', 'DEFAULT_MODEL', 'NIST_API_KEY']
        missing = [name for name in names if not os.environ.get(name, '').strip()]
        if missing:
            raise ValueError('Отсутствуют настройки: ' + ', '.join(missing))
        result = cls(*(os.environ[name].strip() for name in names),
                     max_steps=int(os.getenv('MAX_LLM_STEPS', '40')),
                     max_tools=int(os.getenv('MAX_TOOL_CALLS', '80')),
                     llm_interval=float(os.getenv('LLM_MIN_INTERVAL', '1')),
                     timeout=float(os.getenv('LLM_TIMEOUT', '120')),
                     retries=int(os.getenv('LLM_MAX_RETRIES', '4')),
                     mcp_timeout=float(os.getenv('MCP_TIMEOUT', '3600')))
        if (result.max_steps < 1 or result.max_tools < 1 or result.retries < 0
                or any(not math.isfinite(x) or x <= 0 for x in
                       (result.llm_interval, result.timeout, result.mcp_timeout))):
            raise ValueError('Лимиты и таймауты должны быть положительными; число повторов >= 0')
        return result

    def fingerprint(self):
        data = asdict(self)
        del data['api_key'], data['nist_key']
        data['implementation'] = '0.1.0'
        data['nvd'] = {k: v for k, v in os.environ.items() if k.startswith('NVD_')}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
