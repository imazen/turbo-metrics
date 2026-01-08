; ModuleID = 'shared.cu'
source_filename = "shared.cu"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Shared memory for Malta filter kernels
; 24x24 float array for 16x16 thread blocks with 4-pixel halo
; Flattened to avoid bank conflicts
@MALTA_DIFFS = dso_local addrspace(3) global [576 x float] undef, align 4
