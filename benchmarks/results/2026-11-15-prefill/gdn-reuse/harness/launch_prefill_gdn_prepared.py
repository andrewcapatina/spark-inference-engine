from pathlib import Path
import sys,runpy
r=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'small':'run_prefill_gdn_small.py','ab':'run_prefill_gdn_ab.py','profile':'run_prefill_gdn_profile.py','accuracy':'run_prefill_gdn_accuracy.py'};controller=controllers[kind];path=Path((r/('feedback/prefill-gdn-'+kind+'-run.txt')).read_text().strip());sys.argv=[str(r/controller),'--run',str(path)];runpy.run_path(str(r/controller),run_name='__main__')
