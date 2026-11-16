from pathlib import Path
import sys,json,shutil
import torch
run,mode=sys.argv[1:3];out=Path('/results')/run;out.mkdir(exist_ok=True);d={'status':'complete','mode':mode,'scope':'Installed diagnostic capabilities only; no model or GPU workload','compute_sanitizer':shutil.which('compute-sanitizer'),'scaled_mm_schemas':str(torch.ops.aten._scaled_mm._schemas),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'cudnn_benchmark':torch.backends.cudnn.benchmark,'cudnn_deterministic':torch.backends.cudnn.deterministic,'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction};(out/'results.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d),flush=True)
