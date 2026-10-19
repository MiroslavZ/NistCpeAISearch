import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx2
from openai import APIConnectionError, APIStatusError

from cpe_matcher.agent import Matcher, RESULT_SCHEMA
from cpe_matcher.config import Settings
from cpe_matcher.llm import DeepSeek, IncompleteResponseError, ServiceError, retry_after
from cpe_matcher.mcp_server import create_server
from cpe_matcher.storage import InputRow


def proposal(status='matched', vendor='vendor', product='product', part='a'):
    return {'role': 'assistant', 'content': json.dumps(dict(
        part=part, vendor=vendor, product=product, status=status,
        conclusion='Подтверждено соответствие продукта.'))}


def tool_call(name='search_cpe', arguments=None):
    return {'role': 'assistant', 'content': None, 'tool_calls': [{
        'id': 'call_1', 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments or {'keyword': 'product'})}}]}


class ScriptedLLM:
    def __init__(self, *messages):
        self.messages = iter(messages)
        self.history = []

    async def complete(self, messages, tools):
        self.history.append(copy.deepcopy(messages))
        return next(self.messages)


class FakeSession:
    def __init__(self, *answers):
        self.answers = iter(answers)
        self.calls = []

    async def call_tool(self, name, arguments, **kwargs):
        self.calls.append((name, arguments))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        from mcp.types import CallToolResult, TextContent
        return CallToolResult(is_error=False, content=[TextContent(type='text', text=json.dumps(answer))])


def settings(**overrides):
    return Settings(api_key='test', base_url='https://example.invalid', model='test',
                    nist_key='test', **overrides)


class MatcherTests(unittest.IsolatedAsyncioTestCase):
    async def test_returned_schema_is_rejected_with_concrete_feedback(self):
        llm = ScriptedLLM({'role': 'assistant', 'content': json.dumps(RESULT_SCHEMA)}, proposal())
        session = FakeSession({'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'matched')
        self.assertEqual(len(session.calls), 1)
        self.assertIn('Additional properties', llm.history[1][-1]['content'])

    async def test_truncated_generation_does_not_trigger_json_repair_loop(self):
        llm = SimpleNamespace(complete=AsyncMock(side_effect=IncompleteResponseError(
            'DeepSeek не завершил ответ: finish_reason=length.')))
        answer = await self.make_matcher(llm, FakeSession()).match(self.row)
        self.assertEqual(answer['status'], 'search_incomplete')
        self.assertIn('finish_reason=length', answer['conclusion'])
        self.assertEqual(llm.complete.await_count, 1)
    row = InputRow('vendor', 'product', '1.2')

    async def asyncSetUp(self):
        # Register the production tools without using NVD or opening a session.
        self.tools = await create_server(SimpleNamespace()).list_tools()

    def make_matcher(self, llm, session, **overrides):
        return Matcher(llm, session, self.tools, settings(**overrides))

    async def test_parameter_error_reaches_model_and_allows_corrected_validation(self):
        from mcp.types import CallToolResult, TextContent
        for initial in (proposal(), tool_call('validate_product', {
                'part': 'a', 'vendor': 'vendor', 'product': 'product'})):
            with self.subTest(initial=initial):
                session = SimpleNamespace(call_tool=AsyncMock(side_effect=[
                    CallToolResult(is_error=True, content=[TextContent(type='text',
                        text='NVD HTTP 404: Invalid cpeMatchstring parameter')]),
                    CallToolResult(content=[TextContent(type='text', text=json.dumps({'status': 'valid'}))]),
                ]))
                llm = ScriptedLLM(initial, proposal(product='corrected'))
                answer = await self.make_matcher(llm, session).match(self.row)
                self.assertEqual(answer['status'], 'matched')
                self.assertIn('Invalid cpeMatchstring', llm.history[1][-1]['content'])
                self.assertEqual(session.call_tool.await_count, 2)

    async def test_mcp_service_error_preserves_reason_without_secret(self):
        from mcp.types import CallToolResult, TextContent
        session = SimpleNamespace(call_tool=AsyncMock(return_value=CallToolResult(
            is_error=True, content=[TextContent(type='text',
                text='Error: SERVICE_ERROR: NVD: HTTP 403; message: denied private-key')])) )
        with patch.dict('os.environ', {'NIST_API_KEY': 'private-key'}):
            with self.assertRaises(ServiceError) as raised:
                await self.make_matcher(ScriptedLLM(tool_call()), session).match(self.row)
        self.assertIn('NVD: HTTP 403', str(raised.exception))
        self.assertIn('denied', str(raised.exception))
        self.assertNotIn('private-key', str(raised.exception))

    async def test_rejected_proposal_is_returned_to_model_for_correction(self):
        llm = ScriptedLLM(proposal(product='wrong'), proposal())
        session = FakeSession({'status': 'not_found'}, {'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['cpe_product'], 'product')
        self.assertEqual(len(session.calls), 2)
        feedback = llm.history[1][-1]['content']
        self.assertIn('wrong', feedback)
        self.assertIn('not_found', feedback)

    async def test_nonexistent_product_is_never_accepted(self):
        llm = ScriptedLLM(proposal(), proposal())
        session = FakeSession({'status': 'not_found'})
        answer = await self.make_matcher(llm, session, max_steps=2).match(self.row)
        self.assertEqual(answer['status'], 'search_incomplete')
        self.assertEqual(answer['cpe_product'], '')
        self.assertEqual(len(session.calls), 1)  # Identical checks reuse their response.

    async def test_vendor_only_is_validated_and_has_no_product_or_part(self):
        llm = ScriptedLLM(proposal('vendor_only', product='', part=''))
        session = FakeSession({'status': 'valid', 'vendor': 'canonical_vendor'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(session.calls, [('validate_vendor', {'vendor': 'vendor'})])
        self.assertEqual(answer['cpe_vendor'], 'canonical_vendor')
        self.assertEqual(answer['cpe_part'], '')
        self.assertEqual(answer['cpe_product'], '')

    async def test_parent_remains_parent_after_validation(self):
        llm = ScriptedLLM(tool_call(), proposal('matched_parent'))
        session = FakeSession({'results': []}, {'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'matched_parent')

    async def test_parent_without_search_is_rejected(self):
        llm = ScriptedLLM(proposal('matched_parent'))
        session = FakeSession()
        answer = await self.make_matcher(llm, session, max_steps=1).match(self.row)
        self.assertEqual(answer['status'], 'error')
        self.assertEqual(session.calls, [])

    async def test_empty_product_does_not_call_model(self):
        llm = ScriptedLLM()
        answer = await self.make_matcher(llm, FakeSession()).match(InputRow('vendor', '  ', ''))
        self.assertEqual(answer['status'], 'invalid_input')
        self.assertEqual(llm.history, [])

    async def test_invalid_json_can_be_corrected(self):
        llm = ScriptedLLM({'role': 'assistant', 'content': 'not json'}, proposal())
        answer = await self.make_matcher(llm, FakeSession({'status': 'valid'})).match(self.row)
        self.assertEqual(answer['status'], 'matched')
        self.assertIn('исправленный JSON', llm.history[1][-1]['content'])

    async def test_invalid_final_feedback_identifies_the_actual_error(self):
        missing = json.loads(proposal()['content'])
        del missing['product']
        null_product = json.loads(proposal()['content'])
        null_product['product'] = None
        cases = [
            (json.dumps(missing), "'product' is a required property"),
            (json.dumps(null_product), "Поле product: None is not of type 'string'"),
            ('{bad json}', 'Некорректный JSON:'),
            (r'{"product":"visual_c\+\+"}', 'Invalid \\escape'),
        ]
        for content, reason in cases:
            with self.subTest(content=content):
                llm = ScriptedLLM({'role': 'assistant', 'content': content}, proposal())
                session = FakeSession({'status': 'valid'})
                with self.assertLogs('cpe_matcher.agent', level='WARNING') as logs:
                    answer = await self.make_matcher(llm, session).match(self.row)
                self.assertEqual(answer['status'], 'matched')
                self.assertIn(reason, llm.history[1][-1]['content'])
                self.assertIn(reason, '\n'.join(logs.output))
                self.assertEqual(len(session.calls), 1)

    async def test_fenced_json_is_validated_without_an_extra_model_call(self):
        for fence in ('```json', '```'):
            with self.subTest(fence=fence):
                content = fence + '\n' + proposal()['content'] + '\n```'
                llm = ScriptedLLM({'role': 'assistant', 'content': content})
                session = FakeSession({'status': 'valid'})
                answer = await self.make_matcher(llm, session).match(self.row)
                self.assertEqual(answer['status'], 'matched')
                self.assertEqual(len(llm.history), 1)
                self.assertEqual(session.calls[0][0], 'validate_product')

    async def test_prose_and_trailing_junk_are_not_accepted_as_json(self):
        valid = proposal()['content']
        for content in ('Вот результат: ' + valid, valid + ' trailing junk',
                        '```json\n' + valid + '\n``` trailing junk'):
            with self.subTest(content=content):
                llm = ScriptedLLM({'role': 'assistant', 'content': content}, proposal())
                session = FakeSession({'status': 'valid'})
                answer = await self.make_matcher(llm, session).match(self.row)
                self.assertEqual(answer['status'], 'matched')
                self.assertEqual(len(llm.history), 2)
                self.assertIn('Некорректный JSON:', llm.history[1][-1]['content'])
                self.assertEqual(len(session.calls), 1)

    async def test_fenced_json_still_requires_the_result_schema(self):
        invalid = json.loads(proposal()['content'])
        invalid['product'] = None
        content = '```json\n' + json.dumps(invalid) + '\n```'
        llm = ScriptedLLM({'role': 'assistant', 'content': content}, proposal())
        session = FakeSession({'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'matched')
        self.assertEqual(len(llm.history), 2)
        self.assertIn('Поле product:', llm.history[1][-1]['content'])
        self.assertEqual(len(session.calls), 1)

    async def test_not_found_requires_search(self):
        final = proposal('not_found', part='', vendor='', product='')
        llm = ScriptedLLM(final, tool_call(), final)
        session = FakeSession({'results': []})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'not_found')
        self.assertEqual(len(llm.history), 3)

    async def test_validation_obeys_tool_budget(self):
        llm = ScriptedLLM(tool_call(), proposal())
        session = FakeSession({'results': []})
        answer = await self.make_matcher(llm, session, max_tools=1).match(self.row)
        self.assertEqual(answer['status'], 'search_incomplete')
        self.assertEqual(len(session.calls), 1)

    async def test_technical_failure_stops_instead_of_becoming_not_found(self):
        llm = ScriptedLLM(tool_call())
        with self.assertRaises(ServiceError):
            await self.make_matcher(llm, FakeSession(ConnectionError('offline'))).match(self.row)

    async def test_unknown_tool_is_correctable_without_mcp_request(self):
        llm = ScriptedLLM(tool_call('invented'), proposal())
        session = FakeSession({'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'matched')
        self.assertEqual(len(session.calls), 1)
        self.assertIn('error', json.loads(llm.history[1][-1]['content']))

    async def test_reasoning_is_preserved_in_following_request(self):
        message = tool_call()
        message['reasoning_content'] = 'Reasoning for tool call'
        llm = ScriptedLLM(message, proposal())
        await self.make_matcher(llm, FakeSession({}, {'status': 'valid'})).match(self.row)
        self.assertEqual(llm.history[1][2]['reasoning_content'], message['reasoning_content'])

    async def test_malformed_tool_envelope_is_correctable(self):
        malformed = tool_call()
        del malformed['tool_calls'][0]['id']
        llm = ScriptedLLM(malformed, proposal())
        session = FakeSession({'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'matched')
        self.assertEqual(len(session.calls), 1)
        self.assertFalse(any(m.get('tool_calls') for m in llm.history[1]))

    async def test_incomplete_validation_cannot_be_masked_by_fallback(self):
        for fallback in ('not_found', 'vendor_only', 'matched_parent'):
            with self.subTest(fallback=fallback):
                final = proposal(fallback, part='a' if fallback == 'matched_parent' else '',
                                 product='parent' if fallback == 'matched_parent' else '',
                                 vendor='' if fallback == 'not_found' else 'vendor')
                llm = ScriptedLLM(tool_call(), proposal(), final)
                session = FakeSession({}, {'status': 'incomplete'})
                answer = await self.make_matcher(llm, session).match(self.row)
                self.assertEqual(answer['status'], 'search_incomplete')
                self.assertEqual(answer['cpe_vendor'], '')

    async def test_successful_exact_match_can_follow_incomplete_validation(self):
        llm = ScriptedLLM(proposal(product='candidate'), proposal())
        session = FakeSession({'status': 'incomplete'}, {'status': 'valid'})
        answer = await self.make_matcher(llm, session).match(self.row)
        self.assertEqual(answer['status'], 'matched')


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_completion_preserves_provider_reason(self):
        for reason in ('length', 'content_filter', 'aborted'):
            with self.subTest(reason=reason):
                client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
                    create=AsyncMock(return_value=SimpleNamespace(choices=[
                        SimpleNamespace(finish_reason=reason)])))))
                with patch('cpe_matcher.llm.AsyncOpenAI', return_value=client):
                    llm = DeepSeek(settings())
                with self.assertRaisesRegex(IncompleteResponseError, reason):
                    await llm.complete([], [])
    def retry_transport(self, responses, retries=2):
        create = AsyncMock(side_effect=responses)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        with patch('cpe_matcher.llm.AsyncOpenAI', return_value=client):
            llm = DeepSeek(settings(retries=retries))
        sleep = self.enterContext(patch('cpe_matcher.llm.asyncio.sleep', new_callable=AsyncMock))
        self.enterContext(patch('cpe_matcher.llm.time.monotonic', return_value=100.0))
        self.enterContext(patch('cpe_matcher.llm.random.random', return_value=0.0))
        return llm, create, sleep

    async def test_rate_limit_retries_after_server_delay_then_succeeds(self):
        request = httpx2.Request('POST', 'https://example.invalid/chat/completions')
        response = httpx2.Response(429, headers={'retry-after': '12'}, request=request)
        failure = APIStatusError('rate limit', response=response, body=None)
        message = {'role': 'assistant', 'content': 'ok'}
        success = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason='stop', message=SimpleNamespace(model_dump=lambda **kw: message))])
        llm, create, sleep = self.retry_transport([failure, success])
        self.assertEqual(await llm.complete([], []), message)
        self.assertEqual(create.await_count, 2)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [0.0, 12.0])

    async def test_payment_error_stops_without_retry(self):
        request = httpx2.Request('POST', 'https://example.invalid/chat/completions')
        failure = APIStatusError('payment required',
                                 response=httpx2.Response(402, request=request), body=None)
        llm, create, sleep = self.retry_transport([failure])
        with self.assertRaisesRegex(ServiceError, 'HTTP 402'):
            await llm.complete([], [])
        self.assertEqual(create.await_count, 1)
        self.assertEqual(sleep.await_count, 1)

    async def test_connection_error_stops_after_retry_budget(self):
        request = httpx2.Request('POST', 'https://example.invalid/chat/completions')
        llm, create, sleep = self.retry_transport(
            [APIConnectionError(request=request) for _ in range(3)])
        with self.assertRaisesRegex(ServiceError, 'после повторных попыток'):
            await llm.complete([], [])
        self.assertEqual(create.await_count, 3)
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [0.0, 2.0, 4.0])

    async def test_transport_preserves_reasoning_and_disables_sdk_retries(self):
        message = SimpleNamespace(model_dump=lambda **kw: {
            'role': 'assistant', 'content': None, 'reasoning_content': 'thinking',
            'tool_calls': [], 'refusal': None})
        create = AsyncMock(return_value=SimpleNamespace(choices=[
            SimpleNamespace(finish_reason='stop', message=message)]))
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
                                 close=AsyncMock())
        with patch('cpe_matcher.llm.AsyncOpenAI', return_value=client) as constructor:
            llm = DeepSeek(settings())
        result = await llm.complete([], [])
        self.assertEqual(result['reasoning_content'], 'thinking')
        self.assertNotIn('refusal', result)
        self.assertEqual(constructor.call_args.kwargs['max_retries'], 0)
        await llm.close()
        client.close.assert_awaited_once()

    async def test_empty_response_is_service_error(self):
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(return_value=SimpleNamespace(choices=[])))))
        with patch('cpe_matcher.llm.AsyncOpenAI', return_value=client):
            llm = DeepSeek(settings())
        with self.assertRaises(ServiceError):
            await llm.complete([], [])

    def test_retry_after_handles_seconds_and_invalid_values(self):
        self.assertEqual(retry_after('12'), 12)
        self.assertEqual(retry_after('-1'), 0)
        self.assertEqual(retry_after(None), 0)
        self.assertEqual(retry_after('invalid'), 0)


if __name__ == '__main__':
    unittest.main()
