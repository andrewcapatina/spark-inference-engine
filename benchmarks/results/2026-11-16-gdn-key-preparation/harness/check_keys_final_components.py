"""Serial standalone and integrated checks in one small-tensor worker."""
from pathlib import Path
import sys,runpy,json,shutil
run=sys.argv[1];out=Path('/results')/run
runpy.run_path('/harness/check_gdn_key_prepare.py',run_name='__main__')
standalone=json.loads((out/'results.json').read_text());assert standalone['status']=='complete';shutil.copy2(out/'results.json',out/'standalone-results.json')
runpy.run_path('/harness/check_gdn_keys_integration.py',run_name='__main__')
integrated=json.loads((out/'results.json').read_text());assert integrated['status']=='complete';integrated['standalone_cases']=len(standalone['cases']);integrated['standalone_plan_cases']=len(standalone['planned_multichunk_cases']);(out/'results.json').write_text(json.dumps(integrated,indent=2)+'\n');print('FINAL_RUNTIME_STRIDE_COMPONENT_GATES_EXACT',flush=True)
