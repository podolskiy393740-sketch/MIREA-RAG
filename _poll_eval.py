import paramiko, time, sys

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect('185.23.35.5', username='root', password='Raspberry123!', timeout=15,
          look_for_keys=False, allow_agent=False)

def run(cmd):
    _, out, _ = c.exec_command(cmd, timeout=10)
    out.channel.recv_exit_status()
    return out.read().decode('utf-8', errors='replace').strip()

pid = '1379188'
for i in range(40):
    time.sleep(30)
    alive = run(f'ps -p {pid} -o pid= 2>/dev/null')
    sz = run('stat -c %s /tmp/eval_results.json 2>/dev/null || echo 0')
    print(f't={i*30+30}s  alive={bool(alive.strip())}  result_bytes={sz}', flush=True)
    if not alive.strip():
        break

print('process done')
c.close()
