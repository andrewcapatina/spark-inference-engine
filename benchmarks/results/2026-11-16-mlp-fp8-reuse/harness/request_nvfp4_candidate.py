"""Request implementation from the local Qwen; never execute or edit engine here."""
import json,sys,time,urllib.request,hashlib
from pathlib import Path
run_name,number=sys.argv[1:];prompt=Path('/prompts/nvfp4-task-'+number+'.txt').read_text();out=Path('/workspace/benchmark_runs')/run_name;out.mkdir(exist_ok=True)
handoff_path=Path('/workspace/LOCAL_MODEL_HANDOFF.md')
handoff=handoff_path.read_text() if handoff_path.exists() else ''
assert len(handoff.encode())<=12000,'Handoff exceeds bounded prompt context; summarize it before authoring'
context=('Project history for continuity only; do not repeat completed/rejected work. The current task below defines your scope.\n'+handoff+'\n\nCURRENT TASK:\n'+prompt) if handoff else prompt
payload={'model':'prototype-qwen27b-q8','messages':[{'role':'system','content':'You alone author the native engine implementation. Return only the complete requested Python source, no Markdown or explanation.'},{'role':'user','content':context}],'stream':False,'think':False,'keep_alive':0,'options':{'temperature':0,'seed':0,'num_ctx':8192,'num_predict':3200,'num_thread':4}}
(out/('task-'+number+'-context.json')).write_text(json.dumps({'handoff_present':bool(handoff),'handoff_sha256':hashlib.sha256(handoff.encode()).hexdigest(),'task_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'handoff_bytes':len(handoff.encode())},indent=2)+'\n')
request=urllib.request.Request('http://models:11434/api/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=900) as response:raw=response.read()
(out/('task-'+number+'-response.json')).write_bytes(raw);result=json.loads(raw);candidate=result['message']['content'];(out/('task-'+number+'-candidate.txt')).write_text(candidate)
print(json.dumps({'model':result.get('model'),'done':result.get('done'),'done_reason':result.get('done_reason'),'eval_count':result.get('eval_count'),'lines':len(candidate.splitlines()),'candidate_bytes':len(candidate.encode())}),flush=True)
