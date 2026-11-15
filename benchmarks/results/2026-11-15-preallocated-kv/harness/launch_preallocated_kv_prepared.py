from pathlib import Path
import sys,runpy
r=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'small':'run_preallocated_kv_small.py','decode':'run_preallocated_kv_decode.py','ab':'run_preallocated_kv_ab.py','profile':'run_preallocated_kv_profile.py','accuracy':'run_preallocated_kv_accuracy.py'};controller=controllers[kind];path=Path((r/('feedback/preallocated-kv-'+kind+'-run.txt')).read_text().strip());sys.argv=[str(r/controller),'--run',str(path)];runpy.run_path(str(r/controller),run_name='__main__')
