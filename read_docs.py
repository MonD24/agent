import os, pathlib, sys
_env = pathlib.Path('.env')
for l in _env.read_text().splitlines():
    if '=' in l and not l.startswith('#'):
        k,_,v = l.partition('=')
        os.environ.setdefault(k.strip(), v.strip())

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import StartPlaygroundRequest, EndTrialRequest

client = HarnessServiceClientSync('https://api.bitgn.com')
trial = client.start_playground(StartPlaygroundRequest(benchmark_id='bitgn/pac1-dev', task_id='t23'))
print('Trial:', trial.trial_id)

from bitgn.pcm_connect import PcmRuntimeClientSync
from bitgn.pcm_pb2 import ReadRequest
pcm = PcmRuntimeClientSync(trial.harness_url)

for p in ['/docs/channels/Discord.txt', '/docs/channels/otp.txt', '/docs/channels/AGENTS.MD', '/docs/inbox-task-processing.md']:
    print(f'\n=== {p} ===')
    print(pcm.read(ReadRequest(path=p)).content[:2000])

client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
