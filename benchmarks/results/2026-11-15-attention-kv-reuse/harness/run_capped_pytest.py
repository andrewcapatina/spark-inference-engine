"""Keep original acceptance assertions; cap allocator and release between tests."""
import gc,sys
from pathlib import Path
import torch,pytest
sys.path.insert(0,'/workspace')
torch.cuda.init();torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(56*1024**3/torch.cuda.get_device_properties(0).total_memory)
class Cleanup:
    @pytest.hookimpl(hookwrapper=True,tryfirst=True)
    def pytest_runtest_teardown(self,item,nextitem):
        yield
        gc.collect();torch.cuda.empty_cache()
args=['-q','-rA','-c','/workspace/acceptance/pytest.ini','--confcutdir=/workspace/acceptance','--import-mode=importlib',*sys.argv[1:]]
raise SystemExit(pytest.main(args,plugins=[Cleanup()]))
