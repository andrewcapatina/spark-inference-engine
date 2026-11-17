# Engine development handoff

Current accepted engine: `48d4cc758027f6f5c84d5fd6afa444d4968cdeae` (fused FP8 activation preparation), on top
of e54057e shared-pair reuse and da80ee6 activation fusion. Local Qwen alone
wrote kernel068, FP8Linear.forward069, pair factory071; copied verbatim with
only class indentation. Codex supervised harness/prompts/references/placement.

New qwen_fp8_prepare.prepare_fp8 fuses scale/max/division/cast/pad for CUDA
contiguous BF16 inference rows>=16,width5120/6144. Other inputs and width17408
retain original preparation; decode unchanged. Original GEMMs/weights/scales,
BF16 output/fast accumulation unchanged. No persistent prepared-tensor cache.
Existing eight shared FP8 gate/up pairs still prepare once and use two GEMMs;
factory guards for hooks/overrides/subclasses/devices/shapes remain unchanged.
Reference CUDA scalar /448 follows rounded FP32 reciprocal multiply; tensor
x/scale follows RN division. Preserve clamp, round points, NaN bits, padding,
and disable contraction.9878/32640maxima differ if true division448 is used.

Incremental actual27B warm prefill:256tokens250.9062→241.7116ms save9.1946ms/
3.6646%;1024tokens985.0204→931.0772ms save53.9432ms/5.4764%. All8ABBAblocks
positive perlength; peak memory unchanged.16samples/backend/length,loading and
generation excluded. No output-token speed claim or additive percentages.
Profile217fused kernels replaces1519launches (-1302);233FP8/168NVFP4GEMMs
and GDNstate/key,activation,norm counts unchanged; prefill only,no decode.

Validated359 strict primitive cases under Torch2.13 AND originalTorch2.11,
including32640BF16finite maxima,boundaries,padding,Inf/NaN raw scale/byte bits;
54real-weight integration cases; full actual27B prefill+nextdecode logits/all
caches and8canonical128-ID replies exact;40answers/446NLL exact;all56Qwen/
15Llama protected gates incl original9B/27B numerical/conversation passed.
Report: benchmarks/results/2026-11-16-fp8-preparation-fusion/README.md.
Main pointer: feedback/fp8-prep-main-run.txt. No active implementation task.

Rejected065unsafe padload,066NaN propagation,067unsupported Triton APIs,
070literal optional-import guard retained before local fixes. Previous rejected/
quarantined GDNstate-tail fusion,paired solves,independent chunk batching stay
rejected. Historical generation outlier did not recur; cause remains UNKNOWN.
Keep canonical fixtures and failing-logit capture; source-snapshot/commit
peelback plan remains, including uncommitted prototypes. Never weaken tests.

Previous history: e54057e FP8pair (shortflat,long3.125ms/.315%), da80ee6 MLP
activation (4.87ms/1.90%256;16.59ms/1.65%1024),f1e760f GDNkeys,3139a9b value
casts,ea70e47 gatednorm,dd139d0 norm,2beb3fd FP8max,6dd8d9f QKVreuse. Reports
under benchmarks/results preserve complete evidence. Do not add percentages.

Validation policy: diagnostic/tiny cases differ from real-checkpoint validation.
Require full real27B logits/caches/2x128IDs perbackend/length vs accepted oracle,
frozen40answers/446NLL,all56Qwen/15Llama gates. One large model at a time,
fresh processes for original9B and27B checkpoints. Native allocator32GiB,
primitives8GiB,original gates56GiB;host floor32GiB,new swap<=256MiB. All4
containers stopped exit0/OOMfalse,GPUidle,serviceMainPID0,sameboot,zero newSwap.
Keep STOP/GPU/LARGE holds and use guarded controllers. No packages/downloads/
subagents/unrelated work. Default engine authorship remains LOCAL QWEN ONLY;
earlier user exception was for GDN only. This context file is not training.
