from pathlib import Path
import sys,runpy
r=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'small':'run_attention_kv_small.py','ab':'run_attention_kv_ab.py','profile':'run_attention_kv_profile.py','accuracy':'run_attention_kv_accuracy.py'};controller=controllers[kind];path=Path((r/('feedback/attention-kv-'+kind+'-run.txt')).read_text().strip());sys.argv=[str(r/controller),'--run',str(path)];runpy.run_path(str(r/controller),run_name='__main__')
