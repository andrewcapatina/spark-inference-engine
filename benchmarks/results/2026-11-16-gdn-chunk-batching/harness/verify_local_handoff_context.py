"""Offline request-harness smoke check: no model, container, network or engine writes."""
from pathlib import Path
from unittest.mock import patch
import tempfile,json,hashlib,io,runpy,sys
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');RealPath=Path;seen=[];handoff=(P/'LOCAL_MODEL_HANDOFF.md').read_text()
with tempfile.TemporaryDirectory() as temp:
 root=RealPath(temp);(root/'nvfp4-task-000.txt').write_text('CURRENT_SCOPE_SENTINEL')
 def mapped(value,*args):
  s=str(value)
  if s=='/prompts/nvfp4-task-000.txt':return root/'nvfp4-task-000.txt'
  if s=='/workspace/LOCAL_MODEL_HANDOFF.md':return P/'LOCAL_MODEL_HANDOFF.md'
  if s=='/workspace/benchmark_runs':return root
  return RealPath(value,*args)
 def fake_open(request,timeout):
  data=json.loads(request.data);content=data['messages'][1]['content'];assert handoff in content and content.endswith('CURRENT TASK:\nCURRENT_SCOPE_SENTINEL');assert data['model']=='prototype-qwen27b-q8' and data['keep_alive']==0;seen.append(data)
  return io.BytesIO(json.dumps({'done':True,'done_reason':'stop','message':{'content':'# offline diagnostic response'},'eval_count':0}).encode())
 with patch('pathlib.Path',side_effect=mapped),patch('urllib.request.urlopen',side_effect=fake_open),patch.object(sys,'argv',['request_nvfp4_candidate.py','handoff-context-diagnostic','000']):runpy.run_path(str(R/'acceptance/request_nvfp4_candidate.py'),run_name='__main__')
 assert len(seen)==1;context=json.loads((root/'handoff-context-diagnostic/task-000-context.json').read_text());assert context['handoff_present'] and context['handoff_sha256']==hashlib.sha256(handoff.encode()).hexdigest()
run=RealPath((R/'feedback/prefill-batch-main-run.txt').read_text().strip());(run/'handoff-context-check.json').write_text(json.dumps({'status':'passed','network_calls':0,'model_loaded':False,'engine_writes':False,**context},indent=2)+'\n');print('LOCAL_HANDOFF_INCLUDED_AND_FINGERPRINTED_OFFLINE')
