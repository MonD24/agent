import os, pathlib, sys
_env = pathlib.Path('.env')
for l in _env.read_text().splitlines():
    if '=' in l and not l.startswith('#'):
        k,_,v = l.partition('=')
        os.environ.setdefault(k.strip(), v.strip())

from bitgn.harness_connect import HarnessServiceClientSync
from bitgn.harness_pb2 import StartPlaygroundRequest, EndTrialRequest
from bitgn.vm.pcm_connect import PcmRuntimeClientSync
from bitgn.vm.pcm_pb2 import ReadRequest, SearchRequest

client = HarnessServiceClientSync('https://api.bitgn.com')
trial = client.start_playground(StartPlaygroundRequest(benchmark_id='bitgn/pac1-dev', task_id='t23'))

pcm = PcmRuntimeClientSync(trial.harness_url)

# Read both Anna Fischer contacts and their accounts
for p in ['/contacts/cont_009.json', '/contacts/cont_010.json']:
    print(f'\n=== {p} ===')
    print(pcm.read(ReadRequest(path=p)).content)

# Read the accounts linked to these contacts
for p in ['/accounts/acct_009.json', '/accounts/acct_010.json']:
    print(f'\n=== {p} ===')
    print(pcm.read(ReadRequest(path=p)).content[:500])

# Read Telegram channel doc
print('\n=== /docs/channels/Telegram.txt (first 500 chars) ===')
print(pcm.read(ReadRequest(path='/docs/channels/Telegram.txt')).content[:500])

# Read msg_003 inbox
print('\n=== /inbox/msg_003.txt ===')
print(pcm.read(ReadRequest(path='/inbox/msg_003.txt')).content)

client.end_trial(EndTrialRequest(trial_id=trial.trial_id))
