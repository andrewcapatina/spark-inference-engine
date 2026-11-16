from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1]
if kind=='cleanup':
 sys.argv=[str(R/'check_readd_cleanup.py')];runpy.run_path(str(R/'check_readd_cleanup.py'),run_name='__main__');sys.exit(0)
if kind=='validation':
 sys.argv=[str(R/'finish_readd_validation.py')]+sys.argv[2:];runpy.run_path(str(R/'finish_readd_validation.py'),run_name='__main__');sys.exit(0)
controllers={'baseline':'run_readd_baseline.py','aa':'run_readd_aa.py','primitives':'run_readd_primitives.py','trace':'run_readd_trace.py','inspect':'run_readd_inspect.py','sanitizer':'run_readd_sanitizer.py','sentinel':'run_readd_sentinel.py','legacy':'run_readd_legacy.py','integration':'run_readd_integration.py','ab':'run_readd_ab.py','accuracy':'run_readd_accuracy.py','profile':'run_readd_profile.py'};controller=R/controllers[kind];run=Path((R/f'feedback/prefill-readd-{kind}-run.txt').read_text().strip());sys.argv=[str(controller),'--run',str(run)];runpy.run_path(str(controller),run_name='__main__')
