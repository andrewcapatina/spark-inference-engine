"""Request implementation from the local Qwen; never execute or edit engine here."""
import json,sys,time,urllib.request
from pathlib import Path
run_name,number=sys.argv[1:];prompt=Path('/prompts/nvfp4-task-'+number+'.txt').read_text();out=Path('/workspace/benchmark_runs')/run_name;out.mkdir(exist_ok=True)
payload={'model':'prototype-qwen27b-q8','messages':[{'role':'system','content':'You alone author the native engine implementation. Return only the complete requested Python source, no Markdown or explanation.'},{'role':'user','content':prompt}],'stream':False,'think':False,'keep_alive':0,'options':{'temperature':0,'seed':0,'num_ctx':8192,'num_predict':3200,'num_thread':4}}
request=urllib.request.Request('http://models:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=900) as response:raw=response.read()
(out/('task-'+number+'-response.json')).write_bytes(raw);result=json.loads(raw);candidate=result['message']['content'];(out/('task-'+number+'-candidate.txt')).write_text(candidate)
print(json.dumps({'model':result.get('model'),'done':result.get('done'),'done_reason':result.get('done_reason'),'eval_count':result.get('eval_count'),'lines':len(candidate.splitlines()),'candidate_bytes':len(candidate.encode())}),flush=True)
