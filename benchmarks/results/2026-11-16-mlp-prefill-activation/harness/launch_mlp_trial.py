from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');kind=sys.argv[1];
if kind=='cleanup':
 sys.argv=[str(R/'check_mlp_cleanup.py')];runpy.run_path(str(R/'check_mlp_cleanup.py'),run_name='__main__');sys.exit(0)
if kind=='validation':
 sys.argv=[str(R/'finish_mlp_validation.py')]+sys.argv[2:];runpy.run_path(str(R/'finish_mlp_validation.py'),run_name='__main__');sys.exit(0)
controllers={'audit':'run_mlp_audit.py','integration':'run_mlp_integration.py','compare':'run_mlp_compare.py','ab':'run_mlp_ab.py','accuracy':'run_mlp_accuracy.py','profile':'run_mlp_profile.py'};controller=R/controllers[kind];run=Path((R/f'feedback/mlp-{kind}-run.txt').read_text().strip());sys.argv=[str(controller),'--run',str(run)];runpy.run_path(str(controller),run_name='__main__')
