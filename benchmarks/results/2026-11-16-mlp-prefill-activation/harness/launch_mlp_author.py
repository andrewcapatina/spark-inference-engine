from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');assert sys.argv[1] in ['055','056','057','058','059','060','061','062'];sys.argv=[str(R/'request_mlp_author.py'),sys.argv[1]];runpy.run_path(str(R/'request_mlp_author.py'),run_name='__main__')
