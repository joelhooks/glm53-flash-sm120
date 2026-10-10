#!/usr/bin/env python3
"""Standalone two-session long-context repro client (no private corpus).

Fits N sessions to a prompt-token target by padding with text from any
file you supply, warms each session alone so its prefix is cached, then
sends all sessions at once. Only http://127.0.0.1 endpoints are accepted.

  repro-client.py --padding padding.txt --output runs/x --model glm-5.3-flash

States: warm_failed, wave_clean, wave_failed. A wave request that ends on
the generation time cap counts as alive, not failed. A client failure alone
is not proof of a GPU fault; check the kernel log for Xid lines.

The lab harness that produced our historical runs pads with a frozen copy
of a private repository and a captured agent request template. This client
substitutes a plain padding file and a fixed four-tool schema, so its
prompts differ in content from ours. Use the same target and concurrency.
"""
import argparse
import json
import socket
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

TOOLS = [{'type': 'function', 'function': {
    'name': name, 'description': 'Read-only repository tool.',
    'parameters': {'type': 'object', 'properties': {k: {'type': v} for k, v in fields.items()},
                   'required': list(fields)[:1], 'additionalProperties': False}}}
    for name, fields in [('read', {'path': 'string', 'offset': 'integer', 'limit': 'integer'}),
                         ('bash', {'command': 'string'}),
                         ('grep', {'pattern': 'string', 'path': 'string'}),
                         ('edit', {'path': 'string', 'oldText': 'string', 'newText': 'string'})]]
TASK = 'Read the reference material above and summarise how its main components depend on each other.'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('redirect forbidden')


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())


def check_endpoint(endpoint):
    p = urllib.parse.urlparse(endpoint)
    if p.scheme != 'http' or p.hostname != '127.0.0.1' or p.username or p.password or p.path or p.query or p.fragment:
        raise SystemExit('only explicit http://127.0.0.1:<port> endpoints are allowed')


def post(endpoint, route, payload, timeout=30):
    request = urllib.request.Request(endpoint+route, json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    return OPENER.open(request, timeout=timeout)


def tokenize(endpoint, model, messages):
    with post(endpoint, '/tokenize', {'model': model, 'messages': messages, 'tools': TOOLS, 'add_generation_prompt': True}) as r:
        value = json.load(r)
    return value.get('count', len(value.get('tokens', [])))


def messages_for(padding, chars, salt):
    return [{'role': 'system', 'content': f'Session {salt}. Reference material follows.\n\n'+padding[:chars]},
            {'role': 'user', 'content': TASK}]


def fit(endpoint, model, padding, target, salt):
    """Largest padding prefix whose prompt stays at or under target tokens."""
    lo, hi, best = 0, len(padding), None
    while lo <= hi:
        mid = (lo+hi)//2
        messages = messages_for(padding, mid, salt)
        count = tokenize(endpoint, model, messages)
        if count <= target:
            best, lo = (messages, count), mid+1
        else:
            hi = mid-1
    if best is None:
        raise SystemExit('target is smaller than the empty prompt')
    return best


def call(endpoint, payload, generation_cap=120, arrival_cap=180):
    """Stream one completion with absolute arrival and generation deadlines."""
    result = {'finish_reason': None, 'ttft_s': None, 'error': None, 'timeout': False, 'usage': {}, 'chars': 0}
    begin, done, lock, holder = time.monotonic(), threading.Event(), threading.Lock(), {}

    def watchdog():
        while not done.wait(.05):
            with lock:
                age, first = time.monotonic()-begin, result['ttft_s']
                if age >= arrival_cap or (first is not None and age-first >= generation_cap):
                    result['timeout'] = True
                    try:
                        holder['response'].fp.raw._sock.shutdown(socket.SHUT_RDWR)
                    except (KeyError, AttributeError, OSError):
                        pass
                    return
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        holder['response'] = post(endpoint, '/v1/chat/completions', payload, timeout=arrival_cap)
        for raw in holder['response']:
            if result['timeout'] or raw.strip() == b'data: [DONE]':
                break
            if not raw.startswith(b'data: '):
                continue
            item = json.loads(raw[6:])
            result['usage'] = item.get('usage') or result['usage']
            for choice in item.get('choices', []):
                delta = choice.get('delta', {})
                text = (delta.get('content') or '')+(delta.get('reasoning_content') or delta.get('reasoning') or '')
                with lock:
                    if (text or delta.get('tool_calls')) and result['ttft_s'] is None:
                        result['ttft_s'] = time.monotonic()-begin
                result['chars'] += len(text)
                result['finish_reason'] = choice.get('finish_reason') or result['finish_reason']
        if not result['finish_reason'] and not result['timeout']:
            result['error'] = 'stream ended without finish_reason'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
    finally:
        done.set()
        if 'response' in holder:
            holder['response'].close()
    result['wall_s'] = time.monotonic()-begin
    return result


def alive(row):
    return row['timeout'] or (not row['error'] and row['finish_reason'] is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endpoint', default='http://127.0.0.1:8000')
    ap.add_argument('--model', required=True)
    ap.add_argument('--padding', type=Path, required=True, help='text file, at least about 4 characters per target token')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--target', type=int, default=122880)
    ap.add_argument('--context', type=int, default=204800)
    ap.add_argument('--sessions', type=int, default=2)
    ap.add_argument('--output-cap', type=int, default=20000)
    a = ap.parse_args()
    check_endpoint(a.endpoint)
    a.output.mkdir(parents=True, exist_ok=False)
    padding = a.padding.read_text()
    log = (a.output/'events.jsonl').open('a')

    def event(**fields):
        log.write(json.dumps({'unix_s': time.time(), **fields})+'\n')
        log.flush()
    sessions = []
    for index in range(a.sessions):
        salt = uuid.uuid4().hex
        messages, tokens = fit(a.endpoint, a.model, padding, a.target, salt)
        sessions.append({'salt': salt, 'messages': messages, 'prompt_tokens': tokens})
        event(event='fit', session=index, prompt_tokens=tokens)

    def payload(session, cap):
        return {'model': a.model, 'messages': session['messages'], 'tools': TOOLS, 'tool_choice': 'auto',
                'temperature': 1.0, 'top_p': .95, 'max_tokens': cap, 'stream': True,
                'stream_options': {'include_usage': True}, 'cache_salt': session['salt']}
    status = {'target': a.target, 'sessions': a.sessions, 'prompt_tokens': [s['prompt_tokens'] for s in sessions]}
    for index, session in enumerate(sessions):
        row = call(a.endpoint, payload(session, 128))
        event(event='warm', session=index, **row)
        if not alive(row):
            status['state'] = 'warm_failed'
            break
    else:
        rows = [None]*len(sessions)

        def run(i):
            cap = min(a.output_cap, a.context-sessions[i]['prompt_tokens']-64)
            rows[i] = call(a.endpoint, payload(sessions[i], cap))
        threads = [threading.Thread(target=run, args=(i,)) for i in range(len(sessions))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for i, row in enumerate(rows):
            event(event='wave', session=i, **row)
        status['wave'] = [alive(r) for r in rows]
        status['state'] = 'wave_clean' if all(status['wave']) else 'wave_failed'
    (a.output/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status))


if __name__ == '__main__':
    main()
