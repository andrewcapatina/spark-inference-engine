from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1]
if kind in ['validation','cleanup']:
 script=R/('finish_fp8_pair_validation.py' if kind=='validation' else 'check_fp8_pair_cleanup.py');sys.argv=[str(script)]+sys.argv[2:];runpy.run_path(str(script),run_name='__main__');sys.exit(0)
controllers={'integration':'run_fp8_pair_integration.py','ab':'run_fp8_pair_ab.py','accuracy':'run_fp8_pair_accuracy.py','profile':'run_fp8_pair_profile.py'};controller=R/controllers[kind];run=Path((R/f'feedback/fp8-pair-{kind}-run.txt').read_text().strip());sys.argv=[str(controller),'--run',str(run)];runpy.run_path(str(controller),run_name='__main__')
