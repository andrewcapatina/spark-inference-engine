from pathlib import Path
import sys,runpy
r=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'small':'run_qkv_reuse_small.py','ab':'run_qkv_ab.py','profile':'run_qkv_profile.py','accuracy':'run_qkv_accuracy.py'};controller=controllers[kind];path=Path((r/('feedback/qkv-'+kind+'-run.txt')).read_text().strip());sys.argv=[str(r/controller),'--run',str(path)];runpy.run_path(str(r/controller),run_name='__main__')
