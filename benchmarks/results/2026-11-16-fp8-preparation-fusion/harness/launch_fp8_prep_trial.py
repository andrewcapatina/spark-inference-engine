from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];controllers={'probe':'run_fp8_prep_probe.py','prototype':'run_fp8_prep_prototype.py','integration':'run_fp8_prep_integration.py','ab':'run_fp8_prep_ab.py','accuracy':'run_fp8_prep_accuracy.py','profile':'run_fp8_prep_profile.py','validation':'finish_fp8_prep_validation.py','cleanup':'check_fp8_prep_cleanup.py'};script=R/controllers[kind];
if kind in ('validation','cleanup'):
 sys.argv=[str(script)]+sys.argv[2:];runpy.run_path(str(script),run_name='__main__');raise SystemExit(0)
run=Path((R/f'feedback/fp8-prep-{kind}-run.txt').read_text().strip());sys.argv=[str(script),'--run',str(run)];runpy.run_path(str(script),run_name='__main__')
