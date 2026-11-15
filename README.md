# Moving-point proximity on a GPU

CUDA program that tracks 10,000 points moving on quadratic curves and, at each sample time, reports three points that each have enough neighbors nearby.

The included `output.txt` is a sample run. A new run overwrites it. The three ids at a given time are any three points that meet the criterion. Which three appear depends on thread order, so a later run can list different ids for the same `t`. The number of timestamps that qualify does not depend on that order. The program checks that the GPU and the CPU agree on that number.

## Problem

Each point has an id and five parameters: `x1`, `x2`, `a`, `b`, `c`. At time `t` its position is

```
x = ((x2 - x1) / 2) * sin(t * pi / 2) + (x2 + x1) / 2
y = a * x * x + b * x + c
```

`t` is sampled from -1 to 1 in `tCount` steps (`tCount + 1` samples):

```
t = 2 * i / tCount - 1,  i = 0 .. tCount
```

A point meets the proximity criterion when at least `K` other points lie at Euclidean distance `D` or less. Distances are compared squared, so the inner loop does not call `sqrt`. For each sample, if at least three points qualify, the program writes three of their ids.

## Input

`input.txt`, first line:

```
N K D tCount
```

Then `N` lines:

```
id x1 x2 a b c
```

The sample file is `N = 10000`, `K = 20`, `D = 10`, `tCount = 100` (101 samples). Ids are `0 .. 9999`, each used once.

## Build and run

Requires the CUDA toolkit. `nvcc` links the CUDA runtime itself.

```
nvcc -O3 final_project.cu -o final_project
./final_project
```

Or `make`, then `./final_project`. The program reads `input.txt` from the current directory and writes `output.txt`. It then runs the same search on one CPU core and prints both times, plus whether the two runs found the same number of qualifying timestamps.

On the course cluster, from this directory:

```
sbatch final_project.sbatch
```

That script is only for this cluster. It requests the `rtx3070` partition, loads the `anaconda` module, and activates the `aharoni` environment. On another machine use the `nvcc` command above.

## What the GPU does

Two kernels, one thread per point:

1. `calcCoordsKernel` evaluates `(x, y)` for every point at one `t`.
2. `checkProximityKernel` counts neighbors. Blocks load coordinates through shared memory in tiles of 256, so each coordinate is read from global memory once per block instead of once per pair. A point that reaches `K` neighbors records its id with `atomicAdd` and stops. After three ids are recorded, later tiles stop scanning. Every thread in the block still reaches both `__syncthreads` calls, so the early stop cannot deadlock the block.

Eight CUDA streams overlap consecutive sample times. Each stream has its own coordinate buffer and its own result slots. The host records a CUDA event around the launch loop, including the small copies that bring three ids back. The initial upload of the point table is outside that interval. The CPU timer covers the sequential scan only, after its coordinate buffer has been allocated.

## Precision

The arithmetic is `double`, matching the input values and the formula above. On an RTX 3070, FP64 throughput is about 1/64 of FP32, so this workload does not sit on that GPU's fast path. The CPU comparison uses the same `double` math. Switching to `float` would change which points fall inside distance `D` and would no longer match a double-precision reference.

## Complexity

Until three qualifying points are found, each sample is a brute-force neighbor search: `O(N^2)` distance tests. Shared memory cuts memory traffic. It does not change that complexity. There is no spatial index. For this input every one of the 101 samples does contain at least three qualifying points (hundreds of them at some times), so the early stop applies, but a large fraction of the `N^2` tests still run.

Times are printed by the executable. They are not hard-coded here, because they depend on the GPU and the CPU you run on.
