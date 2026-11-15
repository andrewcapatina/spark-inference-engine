from pathlib import Path
import sys,runpy
r=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'small':'run_prefill_fp8_small.py','ab':'run_prefill_fp8_ab.py','profile':'run_prefill_fp8_profile.py','accuracy':'run_prefill_fp8_accuracy.py'};controller=controllers[kind];path=Path((r/('feedback/prefill-fp8-'+kind+'-run.txt')).read_text().strip());sys.argv=[str(r/controller),'--run',str(path)];runpy.run_path(str(r/controller),run_name='__main__')
