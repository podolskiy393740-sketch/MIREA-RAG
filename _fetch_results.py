import paramiko, json

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('185.23.35.5', username='root', password='Raspberry123!', timeout=15,
          look_for_keys=False, allow_agent=False)
sftp = c.open_sftp()
sftp.get('/tmp/eval_results.json', 'data/eval/last_results.json')
sftp.close()
c.close()

with open('data/eval/last_results.json', encoding='utf-8') as f:
    data = json.load(f)

print(f'Cases: {len(data)}')
fallbacks = sum(1 for r in data if r['needs_human_fallback'])
errors = sum(1 for r in data if r.get('error'))
judge_scores = [r['judge_score'] for r in data if r.get('judge_score') is not None]
faithfulness = [r['faithfulness'] for r in data if r.get('faithfulness') is not None]
answer_rel = [r['answer_relevance'] for r in data if r.get('answer_relevance') is not None]
ctx_recall = [r['context_recall'] for r in data if r.get('context_recall') is not None]

mean = lambda xs: sum(xs)/len(xs) if xs else None
fmt = lambda v: f'{v:.2f}' if v is not None else 'n/a'
ge4 = sum(1 for s in judge_scores if s >= 4)

print(f'Fallback: {fallbacks}/{len(data)}  Errors: {errors}')
print(f'Judge>=4: {ge4}/{len(judge_scores)}  mean={fmt(mean(judge_scores))}')
print(f'Faithfulness:     {fmt(mean(faithfulness))}')
print(f'Answer Relevance: {fmt(mean(answer_rel))}')
print(f'Context Recall:   {fmt(mean(ctx_recall))}')
print()
for r in data:
    status = 'ERR' if r.get('error') else ('FALLBACK' if r['needs_human_fallback'] else 'ok')
    j = r.get('judge_score', '?')
    q = r['question'][:60]
    print(f'[{status:8s}] j={j}  {q}')
    if r.get('error'):
        print(f'           {r["error"][:80]}')
    if r.get('judge_reason') and status != 'ok':
        print(f'           reason: {r["judge_reason"][:80]}')
