from pathlib import Path
import sys,runpy
R=Path('/home/cap/spark-inference-overnight-20261113');assert sys.argv[1] in ['065','066','067','068','069','070','071','072'];sys.argv=[str(R/'request_fp8_prep_author.py'),sys.argv[1]];runpy.run_path(str(R/'request_fp8_prep_author.py'),run_name='__main__')
