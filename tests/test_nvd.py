import json
import unittest
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

from cpe_matcher.cpe import match_string, split_cpe
from cpe_matcher.mcp_server import create_server
from cpe_matcher.nvd import NvdService, NvdServiceError, NvdSettings, RateLimiter


class Clock:
    def __init__(self):
        self.now = 0.

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Response:
    def __init__(self, raw=None, status=200, headers=None):
        self.raw, self.status_code, self.headers = raw, status, headers or {}

    def json(self):
        return self.raw


def cpe(product='widget', deprecated=False, version='1.0', **extra):
    return {'cpe': {'cpeName': f'cpe:2.3:a:vendor:{product}:{version}:*:*:*:*:*:*:*',
                    'deprecated': deprecated, **extra}}


def page(rows, start=0, total=None, kind='products'):
    return Response({kind: rows, 'startIndex': start, 'resultsPerPage': len(rows),
                     'totalResults': len(rows) if total is None else total})


class NvdTests(unittest.TestCase):
    def test_invalid_cpe_404_is_correctable_not_a_service_outage(self):
        service, calls = self.service([Response(status=404, headers={
            'message': 'Invalid cpeMatchstring parameter, see documentation.'}), page([cpe()])])
        with self.assertRaisesRegex(ValueError, 'Invalid cpeMatchstring'):
            service.validate_product('a', 'vendor', 'widget')
        self.assertEqual(len(calls), 1)
        self.assertEqual(service.validate_product('a', 'vendor', 'widget')['status'], 'valid')

    def test_unclassified_404_remains_service_error(self):
        service, _ = self.service([Response(status=404)])
        with self.assertRaises(NvdServiceError):
            service.validate_product('a', 'vendor', 'widget')

    def test_unescaped_plus_is_rejected_locally_and_escaped_plus_survives_transport(self):
        service, calls = self.service([page([cpe(product=r'visual_c\+\+')])])
        with self.assertRaisesRegex(ValueError, 'неэкранированные'):
            service.validate_product('a', 'vendor', 'visual_c++')
        self.assertEqual(len(calls), 0)
        self.assertEqual(service.validate_product('a', 'vendor', r'visual_c\+\+')['status'], 'valid')
        self.assertEqual(calls[0][2]['params']['cpeMatchString'],
                         r'cpe:2.3:a:vendor:visual_c\+\+:*:*:*:*:*:*:*:*')

    def test_http_failure_preserves_status_and_sanitized_server_message(self):
        service, calls = self.service([Response(status=403, headers={
            'message': 'Request denied for offline-key\ntry later'})])
        with self.assertRaises(NvdServiceError) as raised:
            service.search_cpe(keyword='widget')
        message = str(raised.exception)
        self.assertIn('HTTP 403', message)
        self.assertIn('Request denied', message)
        self.assertNotIn('offline-key', message)
        self.assertNotIn('\n', message)
        self.assertEqual(len(calls), 1)

    def test_retry_exhaustion_preserves_last_http_reason(self):
        service, calls = self.service([
            Response(status=503), Response(status=429, headers={'message': 'Rate limit exceeded'})
        ], max_retries=1)
        with self.assertRaises(NvdServiceError) as raised:
            service.search_cpe(keyword='widget')
        self.assertIn('HTTP 429', str(raised.exception))
        self.assertIn('Rate limit exceeded', str(raised.exception))
        self.assertIn('(2)', str(raised.exception))
        self.assertEqual(len(calls), 2)

    def service(self, responses, **settings):
        clock, calls = Clock(), []
        responses = iter(responses)

        def request(url, **kwargs):
            calls.append((clock.time(), url, kwargs))
            return next(responses)

        service = NvdService(NvdSettings('offline-key', **settings), request,
                             clock=clock.time, sleep=clock.sleep)
        return service, calls

    def test_active_any_version_after_deprecated_page(self):
        service, calls = self.service([
            page([cpe(deprecated=True)], total=2), page([cpe(version='9.0')], start=1, total=2),
        ], page_size=1)
        result = service.validate_product('a', 'vendor', 'widget')
        self.assertEqual(result['status'], 'valid')
        self.assertIn(':9.0:', result['evidence'][0]['cpe'])
        self.assertEqual([c[2]['params']['startIndex'] for c in calls], [0, 1])
        self.assertGreaterEqual(calls[1][0] - calls[0][0], 1)
        self.assertEqual(service.validate_product('a', 'vendor', 'widget'), result)
        self.assertEqual(len(calls), 2)

    def test_partial_deprecated_page_is_incomplete(self):
        service, _ = self.service([page([cpe(deprecated=True)], total=2)], max_pages=1)
        self.assertEqual(service.validate_product('a', 'vendor', 'widget')['status'], 'incomplete')

    def test_deprecated_complete_and_replacement_not_auto_accepted(self):
        replacement = {'cpeName': 'cpe:2.3:a:other:unrelated:*:*:*:*:*:*:*:*'}
        service, _ = self.service([page([cpe(deprecated=True, deprecatedBy=[replacement])])])
        result = service.validate_product('a', 'vendor', 'widget')
        self.assertEqual(result['status'], 'deprecated_only')
        self.assertEqual(result['replacements'], [replacement])

    def test_similar_name_cannot_validate_exact_triple(self):
        service, _ = self.service([page([cpe(product='widget_pro')])])
        self.assertEqual(service.validate_product('a', 'vendor', 'widget')['status'], 'absent')

    def test_vendor_title_is_not_vendor_component(self):
        service, _ = self.service([page([cpe(titles=[{'title': 'Other company'}])])])
        self.assertEqual(service.validate_vendor('other')['status'], 'absent')

    def test_retry_after_and_pagination_share_request_limiter(self):
        service, calls = self.service([
            Response(status=429, headers={'Retry-After': '12'}),
            page([cpe(deprecated=True)], total=2), page([cpe()], start=1, total=2),
        ])
        self.assertEqual(service.validate_product('a', 'vendor', 'widget')['status'], 'valid')
        self.assertGreaterEqual(calls[1][0], 12)
        self.assertGreaterEqual(calls[2][0] - calls[1][0], 1)

    def test_auth_error_not_retried_or_cached_as_absence(self):
        service, calls = self.service([Response(status=403), page([cpe()])])
        with self.assertRaisesRegex(NvdServiceError, 'SERVICE_ERROR:'):
            service.validate_product('a', 'vendor', 'widget')
        self.assertEqual(len(calls), 1)
        self.assertEqual(service.validate_product('a', 'vendor', 'widget')['status'], 'valid')

    def test_retry_budget_stops_service(self):
        service, calls = self.service([Response(status=503), Response(status=503)], max_retries=1)
        with self.assertRaises(NvdServiceError):
            service.validate_vendor('vendor')
        self.assertEqual(len(calls), 2)

    def test_empty_page_cannot_hide_remaining_results(self):
        service, _ = self.service([page([], total=5)])
        with self.assertRaises(NvdServiceError):
            service.validate_vendor('vendor')

    def test_missing_metadata_is_service_failure(self):
        service, _ = self.service([Response({'products': []})])
        with self.assertRaises(NvdServiceError):
            service.validate_vendor('vendor')

    def test_search_groups_versions_and_retains_deprecation(self):
        service, _ = self.service([page([cpe(deprecated=True), cpe(), cpe(version='2')])])
        result = service.search_cpe(keyword='Widget & name=value')
        self.assertFalse(result['incomplete'])
        self.assertEqual(len(result['groups']), 1)
        self.assertEqual({e['deprecated'] for e in result['groups'][0]['evidence']}, {True, False})

    def test_keyword_cannot_inject_nvd_query_parameters(self):
        service, calls = self.service([page([])])
        service.search_cpe(keyword='a & resultsPerPage=9000')
        params = calls[0][2]['params']
        self.assertEqual(params['keywordSearch'], 'a & resultsPerPage=9000')
        self.assertEqual(params['resultsPerPage'], '1000')

    def test_cve_context_preserves_environment_and_operator(self):
        configurations = [{'operator': 'AND', 'nodes': [{'cpeMatch': [
            {'vulnerable': False, 'criteria': 'cpe:2.3:o:vendor:os:*:*:*:*:*:*:*:*'}]}]}]
        service, _ = self.service([page([{'cve': {'id': 'CVE-2020-0001', 'configurations': configurations}}], kind='vulnerabilities')])
        result = service.search_cve('widget')
        self.assertEqual(result['items'][0]['configurations'], configurations)

    def test_escaped_cpe_colon_and_backslash(self):
        name = match_string('a', r'ven\:dor', r'pro\\duct')
        self.assertEqual(split_cpe(name)[:3], ('a', r'ven\:dor', r'pro\\duct'))
        with self.assertRaises(ValueError):
            match_string('a', 'vendor', 'wid*')

    def test_canonical_components_and_case_insensitive_cpe_matching(self):
        service, _ = self.service([page([cpe(product=r'widget\:extra')])])
        result = service.validate_product('a', 'VENDOR', r'widget\:extra')
        self.assertEqual(result['status'], 'valid')
        self.assertEqual(result['vendor'], 'vendor')
        self.assertEqual(result['product'], r'widget\:extra')

    def test_cve_configuration_truncation_is_explicit(self):
        matches = [{'vulnerable': False, 'criteria': 'cpe:2.3:o:vendor:os:*:*:*:*:*:*:*:*'}] * 150
        service, _ = self.service([page([{'cve': {'id': 'CVE-2020-0001',
            'configurations': [{'operator': 'AND', 'nodes': [{'cpeMatch': matches}]}]}}], kind='vulnerabilities')])
        result = service.search_cve('widget')
        self.assertTrue(result['incomplete'])
        self.assertTrue(result['configurations_truncated'])
        self.assertEqual(result['items'][0]['configurations'][0]['operator'], 'AND')

    def test_settings_repr_does_not_contain_api_key(self):
        self.assertNotIn('secret-value', repr(NvdSettings('secret-value')))

    def test_hard_sliding_window_even_if_spacing_disabled_in_limiter_test(self):
        clock = Clock()
        limiter = RateLimiter(0, clock.time, clock.sleep)
        starts = []
        for _ in range(120):
            limiter.wait()
            starts.append(clock.time())
        for end in starts:
            self.assertLessEqual(sum(end - 30 < start <= end for start in starts), 50)


class McpServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_raw_name_is_corrected_to_hypothesis_before_nvd_search(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from mcp.types import CallToolResult, TextContent
        from cpe_matcher.agent import Matcher
        from cpe_matcher.config import Settings
        from cpe_matcher.storage import InputRow

        service, calls = NvdTests().service([page([cpe()]), page([cpe()])])
        server = create_server(service)

        async def call_tool(name, arguments, **kwargs):
            try:
                return await server.call_tool(name, arguments)
            except ToolError as exc:
                return CallToolResult(is_error=True, content=[TextContent(type='text', text=str(exc))])

        def search(identifier, vendor, product):
            return {'role': 'assistant', 'tool_calls': [{
                'id': identifier, 'type': 'function', 'function': {
                    'name': 'search_cpe', 'arguments': json.dumps({
                        'part': 'a', 'vendor': vendor, 'product': product})}}]}

        llm = SimpleNamespace(complete=AsyncMock(side_effect=[
            search('raw', 'Производитель', 'Название программы'),
            search('hypothesis', 'vendor', 'widget'),
            {'role': 'assistant', 'content': json.dumps({
                'part': 'a', 'vendor': 'vendor', 'product': 'widget',
                'status': 'matched', 'conclusion': 'Продукт подтверждён словарём.'})},
        ]))
        matcher = Matcher(llm, SimpleNamespace(call_tool=call_tool), await server.list_tools(),
                          Settings('offline', 'https://example.invalid', 'offline', 'offline'))
        result = await matcher.match(InputRow('Производитель', 'Название программы', '1'))
        self.assertEqual(result['status'], 'matched')
        self.assertEqual(len(calls), 2)  # Search and mandatory validation; no HTTP for raw fields.
        self.assertTrue(all(call[2]['params']['cpeMatchString'] ==
                            'cpe:2.3:a:vendor:widget:*:*:*:*:*:*:*:*' for call in calls))
        messages = llm.complete.call_args.args[0]
        feedback = next(message for message in messages if message.get('tool_call_id') == 'raw')
        self.assertIn('гипотезу', json.loads(feedback['content'])['action'])

    async def test_real_sdk_success_and_schema(self):
        service, _ = NvdTests().service([page([cpe()])])
        server = create_server(service)
        tools = await server.list_tools()
        tool = next(item for item in tools if item.name == 'validate_product')
        self.assertEqual(set(tool.input_schema['required']), {'part', 'vendor', 'product'})
        result = await server.call_tool('validate_product', {'part': 'a', 'vendor': 'vendor', 'product': 'widget'})
        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content[0].text)['status'], 'valid')

    async def test_real_sdk_preserves_service_error_marker(self):
        service, _ = NvdTests().service([Response(status=403)])
        server = create_server(service)
        with self.assertRaises(ToolError) as raised:
            await server.call_tool('validate_vendor', {'vendor': 'vendor'})
        self.assertNotIsInstance(raised.exception, UnexpectedToolError)
        self.assertIn('SERVICE_ERROR:', str(raised.exception))
        self.assertNotIn('offline-key', str(raised.exception))

    async def test_real_sdk_surfaces_invalid_query_as_tool_error(self):
        service, calls = NvdTests().service([])
        with self.assertRaises(ToolError) as raised:
            await create_server(service).call_tool('search_cpe', {'keyword': ''})
        self.assertNotIsInstance(raised.exception, UnexpectedToolError)
        self.assertNotIn('SERVICE_ERROR:', str(raised.exception))
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
