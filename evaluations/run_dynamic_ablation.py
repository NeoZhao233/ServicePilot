"""Isolated, offline engineering ablation. Never imports safety ablations into production."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
GROUPS = {
    'B0': 'Frozen original runtime',
    'B1': 'Frozen original + observation-only structured task trace',
    'B2': 'Dynamic template plans + durable backend; no outcome reconciliation or version predicate',
    'B3': 'B2 + outcome reconciliation',
    'B4': 'B3 + transactional scoped authorization predicate (full implementation)',
    'SAFE': 'New safety/recovery backend + original workflow; no dynamic planner',
}
CASES = ['goal_change', 'goal_with_confirmation', 'response_lost', 'status_between_turns',
         'status_at_write', 'amount_between_turns', 'amount_at_write', 'unknown', 'simple_query']


def worker(group, scenario):
    from servicepilot.agent import ServiceAgent
    from servicepilot.graph_runtime import LangGraphServiceRuntime
    from servicepilot.models import OrderStatus, ToolResult
    from servicepilot.repository import InMemoryCommerceRepository
    from servicepilot.retrieval import PolicyRetriever

    old = group in {'B0', 'B1'}
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / 'business.sqlite3'
        if old:
            repo = InMemoryCommerceRepository.from_json(ROOT / 'data/orders.json')
        else:
            from servicepilot.repository import SQLiteCommerceRepository
            repo = SQLiteCommerceRepository.from_json(ROOT / 'data/orders.json', database=database)
        if group in {'B2', 'B3'}:
            from servicepilot.policy_guard import PolicyGuard
            # Test-process-only ablation. Keep original confirmation, ownership and eligibility.
            PolicyGuard.authorization_valid = staticmethod(
                lambda grant, context, **kw: bool(grant['confirmed'] and not grant['invalidated']))
        if group == 'B2':
            from servicepilot.recovery import RecoveryPolicy
            def no_recovery(repository, *, operation_id, user_id, events):
                repository.freeze_operation(operation_id, user_id, 'ablation: no reconciliation')
                return ToolResult(False, {'code': 'outcome_unknown', 'outcome': 'unknown'},
                                  '结果尚未确认，需人工核对')
            RecoveryPolicy.reconcile_refund = staticmethod(no_recovery)

        agent = ServiceAgent(repo, PolicyRetriever.from_json(ROOT / 'data/policies.json'))
        runtime = LangGraphServiceRuntime(agent) if old else LangGraphServiceRuntime(
            agent, dynamic=group != 'SAFE')
        writes = []
        original = repo.create_refund_request

        def change(**kwargs):
            if old:
                repo._orders['ORD-1001'] = replace(repo._orders['ORD-1001'], **kwargs)
            else:
                repo.update_order('ORD-1001', **kwargs)

        def submit(**kwargs):
            writes.append({'order_id': kwargs['order_id'], 'user_id': kwargs['user_id']})
            if scenario == 'status_at_write':
                change(status=OrderStatus.SHIPPED)
            if scenario == 'amount_at_write':
                change(amount=599)
            if scenario == 'unknown':
                raise TimeoutError('injected before observable outcome')
            result = original(**kwargs)
            if scenario == 'response_lost':
                raise TimeoutError('injected response loss after commit')
            return result

        repo.create_refund_request = submit
        traces = []
        started = time.perf_counter()
        first = runtime.chat(user_id='USER-001', message=(
            'ORD-1003 物流' if scenario == 'simple_query' else '退掉 ORD-1001'))
        traces.append({'message': first.message, 'events': [asdict(e) for e in first.events]})
        structured = [{'goal': first.session.intent.value, 'pending': first.session.pending_action}]
        if scenario == 'status_between_turns':
            change(status=OrderStatus.SHIPPED)
        if scenario == 'amount_between_turns':
            change(amount=599)
        message = {'goal_change': '改成换货',
                   'goal_with_confirmation': '确认改成换货'}.get(scenario, '确认')
        response = first
        if scenario != 'simple_query':
            response = runtime.chat(user_id='USER-001', message=message,
                                    session_id=first.session.session_id)
            traces.append({'message': response.message, 'events': [asdict(e) for e in response.events]})
            structured.append({'goal': response.session.intent.value,
                               'pending': response.session.pending_action})
        latency = time.perf_counter() - started
        if old:
            refunds, tickets = list(repo._refund_requests.values()), list(repo._tickets.values())
            task = None
            ledger = None
            order = asdict(repo._orders['ORD-1001'])
        else:
            import sqlite3
            with sqlite3.connect(database) as conn:
                conn.row_factory = sqlite3.Row
                ledger = {name: [dict(row) for row in conn.execute('SELECT * FROM commerce_' + name)]
                          for name in ['refunds', 'tickets', 'operations', 'attempts', 'orders']}
            refunds, tickets = ledger['refunds'], ledger['tickets']
            task = repo.load_task(first.session.session_id)
            order = repo.get_order(order_id='ORD-1001', user_id='USER-001').data
        success = False
        if scenario in {'goal_change', 'goal_with_confirmation'}:
            success = len(refunds) == 0 and len(tickets) == 1
        elif scenario == 'response_lost':
            success = len(refunds) == 1 and len(writes) == 1 and '已提交' in response.message
        elif scenario == 'unknown':
            success = not refunds and len(writes) == 1 and '人工' in response.message
        elif scenario == 'simple_query':
            success = '运输中' in response.message and not refunds
        else:
            success = not refunds
        unsafe = int(bool(refunds) and scenario in {'goal_change', 'goal_with_confirmation',
            'status_between_turns', 'status_at_write', 'amount_between_turns', 'amount_at_write'})
        return {
            'group': group, 'scenario': scenario, 'success': success,
            'latency_ms': round(latency * 1000, 3), 'refund_calls': len(writes),
            'refund_records': len(refunds), 'duplicate_refunds': max(0, len(refunds) - 1),
            'invalid_authorization_or_precondition': unsafe,
            'old_goal_execution': int(bool(refunds) and scenario.startswith('goal')),
            'manual_handoff': '人工' in response.message,
            'tool_calls': sum(int(e['detail'].get('attempts', 1)) for t in traces
                              for e in t['events'] if e['event_type'] in {'tool', 'retrieval'}),
            'llm_tokens': 0, 'planner': 'offline template; no online LLM',
            'traces': traces, 'write_calls': writes, 'final_order': order,
            'ledger': ledger, 'task': task,
            'structured_trace': structured if group == 'B1' else None,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=2)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'evaluations/results/dynamic_ablation.json')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(*args.worker), ensure_ascii=False))
        return
    results = []
    with tempfile.TemporaryDirectory() as frozen:
        with ZipFile(ROOT / 'evaluations/baselines/original_runtime.zip') as archive:
            archive.extractall(frozen)
        for group in GROUPS:
            for scenario in CASES:
                for repeat in range(args.repeat):
                    env = dict(os.environ, PYTHONPATH=frozen if group in {'B0', 'B1'}
                               else str(ROOT / 'src'))
                    output = subprocess.check_output([sys.executable, str(Path(__file__).resolve()),
                        '--worker', group, scenario], env=env, text=True)
                    row = json.loads(output)
                    row['repeat'] = repeat
                    results.append(row)
    summary = {}
    for group in GROUPS:
        rows = [r for r in results if r['group'] == group]
        changed = [r for r in rows if r['scenario'].startswith('goal')]
        recoveries = [r for r in rows if r['scenario'] == 'response_lost']
        summary[group] = {
            'cases': len(rows), 'successes': sum(r['success'] for r in rows),
            'goal_change_success_rate': statistics.mean(r['success'] for r in changed),
            'response_loss_recovery_rate': statistics.mean(r['success'] for r in recoveries),
            'duplicate_refunds': sum(r['duplicate_refunds'] for r in rows),
            'invalid_operations': sum(r['invalid_authorization_or_precondition'] for r in rows),
            'old_goal_execution': sum(r['old_goal_execution'] for r in rows),
            'mean_tool_calls': statistics.mean(r['tool_calls'] for r in rows),
            'mean_latency_ms': statistics.mean(r['latency_ms'] for r in rows),
            'manual_handoff_rate': statistics.mean(r['manual_handoff'] for r in rows),
            'llm_tokens': sum(r['llm_tokens'] for r in rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'groups': GROUPS, 'summary': summary, 'results': results,
        'limitations': ['Offline fixed scenarios, not a general success-rate benchmark.',
                       'B1 is observation-only state instrumentation.',
                       'B2 changes business backend as well as planning; B2-B1 is confounded.',
                       'All groups retain original ownership and confirmation constraints.',
                       'Only B2/B3 test processes disable the NEW version predicate.',
                       'B0/B1 have no business database; ledger=null.',
                       'SAFE is the fair baseline for dynamic planning incremental value.']},
                                     ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
