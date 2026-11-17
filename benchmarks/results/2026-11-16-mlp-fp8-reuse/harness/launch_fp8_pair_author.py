from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');assert sys.argv[1] in ['062','063','064','065','066','067'];sys.argv=[str(R/'request_fp8_pair_author.py'),sys.argv[1]];runpy.run_path(str(R/'request_fp8_pair_author.py'),run_name='__main__')
