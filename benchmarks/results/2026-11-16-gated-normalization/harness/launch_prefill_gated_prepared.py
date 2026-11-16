from pathlib import Path
import sys,runpy
r=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'baseline':'run_prefill_gated_profile.py','small':'run_prefill_gated_small.py','integration':'run_prefill_gated_integration.py','ab':'run_prefill_gated_ab.py','profile':'run_prefill_gated_candidate_profile.py','accuracy':'run_prefill_gated_accuracy.py'};controller=controllers[kind];path=Path((r/('feedback/prefill-gated-'+kind+'-run.txt')).read_text().strip());sys.argv=[str(r/controller),'--run',str(path)];runpy.run_path(str(r/controller),run_name='__main__')
