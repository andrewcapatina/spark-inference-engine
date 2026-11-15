"""CPU-only paired accuracy comparison against criteria fixed before outputs."""
from pathlib import Path
import sys,json,math,hashlib
R=Path('/home/cap/spark-inference-overnight-20261113');P=Path('/home/cap/Projects/spark-inference-engine');B=R/'feedback/nvfp4-reference'

def compare(native,reference,suite):
    assert native['status']==reference['status']=='complete'
    assert native['suite_sha256']==reference['suite_sha256']
    assert native['scoring_sha256']==reference['scoring_sha256']
    expected=[c['id'] for c in suite['cases']]
    assert [c['id'] for c in native['cases']]==[c['id'] for c in reference['cases']]==expected
    regressions=[];improvements=[];both_failed=[];categories={}
    for n,r in zip(native['cases'],reference['cases']):
        assert n['prompt_token_ids']==r['prompt_token_ids'],n['id']
        bucket=categories.setdefault(n['category'],{'native_passed':0,'reference_passed':0,'total':0});bucket['total']+=1;bucket['native_passed']+=int(n['passed']);bucket['reference_passed']+=int(r['passed'])
        if r['passed'] and not n['passed']:regressions.append(n['id'])
        if n['passed'] and not r['passed']:improvements.append(n['id'])
        if not n['passed'] and not r['passed']:both_failed.append(n['id'])
    assert len(native['corpus'])==len(reference['corpus'])==len(suite['corpus'])
    corpus=[]
    for n,r in zip(native['corpus'],reference['corpus']):
        assert n['id']==r['id'] and n['token_ids']==r['token_ids'];assert len(n['token_nll'])==len(r['token_nll'])==len(n['token_ids'])-1
        corpus.append({'id':n['id'],'tokens':n['scored_tokens'],'native_perplexity':n['perplexity'],'reference_perplexity':r['perplexity'],'ratio':n['perplexity']/r['perplexity']})
    n_tokens=sum(c['scored_tokens'] for c in native['corpus']);r_tokens=sum(c['scored_tokens'] for c in reference['corpus']);assert n_tokens==r_tokens
    nppl=math.exp(sum(c['sum_nll'] for c in native['corpus'])/n_tokens);rppl=math.exp(sum(c['sum_nll'] for c in reference['corpus'])/r_tokens)
    accuracy=sum(c['passed'] for c in native['cases'])/len(expected);truncated=[{'runtime':runtime,'id':c['id']} for runtime,d in [('native',native),('reference',reference)] for c in d['cases'] if c['truncated']]
    limits=suite['criteria_fixed_before_outputs'];checks={'minimum_native_task_accuracy':accuracy>=limits['minimum_native_task_accuracy'],'no_new_task_failures':len(regressions)<=limits['maximum_native_only_failures_vs_reference'],'perplexity_ratio_within_bound':nppl/rppl<=limits['maximum_native_to_reference_tokenweighted_perplexity_ratio'],'no_length_truncation':not truncated if limits['all_requests_must_finish_without_length_truncation'] else True}
    return {'status':'screen_passed' if all(checks.values()) else 'screen_failed','scope':suite['scope'],'criteria':limits,'checks':checks,'suite_sha256':native['suite_sha256'],'native_passed':sum(c['passed'] for c in native['cases']),'reference_passed':sum(c['passed'] for c in reference['cases']),'total':len(expected),'categories':categories,'native_only_failures':regressions,'reference_only_failures':improvements,'both_failed':both_failed,'length_truncations':truncated,'corpus_tokens':n_tokens,'native_tokenweighted_perplexity':nppl,'reference_tokenweighted_perplexity':rppl,'perplexity_ratio':nppl/rppl,'corpus':corpus,'engine_changes':False,'caveat':'40original fixed tasks and4 short original prose docs are a preliminary screening set, not a standard benchmark, statistical noninferiority study, or broad quality certification.'}
if __name__=='__main__':
    nrun,rrun=sys.argv[1:3];native=json.loads((B/nrun/'results.json').read_text());reference=json.loads((B/rrun/'results.json').read_text());suite_path=P/'benchmarks/accuracy/nvfp4-smoke-v1.json';suite=json.loads(suite_path.read_text());assert hashlib.sha256(suite_path.read_bytes()).hexdigest()==native['suite_sha256'];result=compare(native,reference,suite);result.update(native_run=nrun,reference_run=rrun);path=R/'runs'/rrun/'accuracy-comparison.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
