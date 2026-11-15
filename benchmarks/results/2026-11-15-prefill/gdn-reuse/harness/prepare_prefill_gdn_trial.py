from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,sys
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');kind=sys.argv[1];assert kind in ('small','ab','profile','accuracy')
run=R/'runs'/('prefill-gdn-'+kind+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'));run.mkdir();plan={'profiling_authorized':True,'one_model_at_a_time':True,'mode':'native' if kind in ('profile','accuracy') else 'candidate','boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'inactive_trace_run':'quant-profile-native-20261115T041946Z','purpose':'User authorized prefill-only GDN common-subexpression reuse; primary metrics prompt tokens/sec and prefill latency; '+kind}
if kind=='accuracy':
 plan['suite_sha256']=hashlib.sha256((P/'benchmarks/accuracy/nvfp4-smoke-v1.json').read_bytes()).hexdigest();plan['scoring_sha256']=hashlib.sha256((P/'tools/accuracy_scoring.py').read_bytes()).hexdigest()
(run/'explicit-post-update-plan.json').write_text(json.dumps(plan,indent=2)+'\n');(R/('feedback/prefill-gdn-'+kind+'-run.txt')).write_text(str(run)+'\n');print(run)
