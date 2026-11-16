"""Preserve all acceptance gates; isolate large checkpoints in fresh processes."""
import hashlib,json,shutil
from pathlib import Path
from nvfp4_update_runtime import Runtime,R,P,C
run=Path((R/'feedback/prefill-readd-main-run.txt').read_text().strip())
prior_path=run/'protected-prefill-readd-regression-status.json'
prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
passed=[]
if prior:
    assert prior['status']=='failed', 'Refuse duplicate or already completed protected validation'
    shutil.copy2(prior_path,run/'protected-before-exec-cache-recovery-status.json')
    passed=[x for x in prior['completed_commands'] if x['exit_code']==0]
    for command in prior['completed_commands']:
        if command['exit_code']:
            for suffix in ['.log','-host-memory.jsonl']:
                source=run/(command['label']+suffix)
                if source.exists():shutil.copy2(source,run/(command['label']+'-before-exec-cache-recovery'+suffix))
rt=Runtime(run,'protected-prefill-readd-regression',swap_policy='growth');error=None
rt.state['completed_commands']=passed;rt.state['harness_recovery']='Writable per-container /tmp Triton cache; prior passed unchanged-source stages reused';rt.save()
try:
    manifest=json.loads((run/'manifest.json').read_text())
    expected=manifest['current_engine_sha256']
    assert expected=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
    if not any(x['label']=='generation-budget-repository' for x in passed):rt.execute('generation-budget-repository',['docker','exec','-w','/workspace',C,'python3','-m','unittest','discover','-s','tests','-p','test_generation_budget.py'],120)
    root='/workspace/acceptance/test_qwen_contract.py'
    selections=[('qwen-small',[root,'-k','not real9b and not real27b'])]
    head=root+'::test_qwen_gguf_head_layout'
    selections += [('qwen-real9b',[root+'::test_qwen_real9b']),('qwen-real9b-conversation',[head,root+'::test_qwen_real9b_conversation']),('qwen-real27b',[head,root+'::test_qwen_real27b']),('qwen-real27b-conversation',[root+'::test_qwen_real27b_conversation']),('llama-protected',['/workspace/acceptance/test_contract.py'])]
    for label,targets in selections:
        if any(x['label']==label for x in passed):continue
        if label.startswith('qwen-real'):
            # File-scoped advice only for inactive experiment model files; no global cache flush.
            import os,subprocess
            assert subprocess.check_output(['docker','exec',C,'nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()==''
            identity=json.loads((R/'model-runtime.json').read_text());nas=Path('/mnt/nas-aimodels/huggingface')
            paths=[nas/identity['source_repo']/identity['source_file']]+list((nas/'xtools-at/DeepSeek-V4-Pro-Qwen3.5-9B-UNCENSORED-heretic').glob('*.safetensors'))+list((nas/'unsloth/Qwen3.8-27B-NVFP4').glob('*.safetensors'))
            paths+=list((R/'feedback/nvfp4-reference/quant-profile-native-20261115T041946Z').glob('*-native-trace.json'))
            profile_pointer=R/'feedback/prefill-readd-profile-run.txt'
            if profile_pointer.exists():
                completed_profile=R/'feedback/nvfp4-reference'/Path(profile_pointer.read_text().strip()).name
                if (completed_profile/'results.json').exists() and json.loads((completed_profile/'results.json').read_text())['status']=='complete':paths+=list(completed_profile.glob('*-native-trace.json'))
            advised=[]
            for path in paths:
                if not path.exists():continue
                stat=path.stat();fd=os.open(path,os.O_RDONLY)
                try:os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED)
                finally:os.close(fd)
                assert (stat.st_size,stat.st_mtime_ns)==(path.stat().st_size,path.stat().st_mtime_ns)
                advised.append(str(path))
            (run/(label+'-inactive-cache-advice.json')).write_text(json.dumps(advised,indent=2))
            probe='import torch;torch.cuda.init();print(torch.cuda.mem_get_info()[0])'
            available=int(subprocess.check_output(['docker','exec',C,'python3','-c',probe],text=True));required=56 if '9b' in label else 75;assert available>required*1024**3,'Insufficient model-specific CUDA headroom for '+label

        rt.execute(label,['docker','exec','-e','TRITON_CACHE_DIR=/workspace/benchmark_runs/gdn-outlier-readd-20261116T145716Z/triton-legacy-cache',C,'timeout','--kill-after=10','1200','python3','-I','/workspace/acceptance/run_capped_pytest.py',*targets],1200)
    assert expected=={str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (P/'spark_engine').glob('*.py')}
except BaseException as e:error=e;raise
finally:rt.close(error)
