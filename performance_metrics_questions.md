Performance Metrics questions:

- At some point, it was mentioned that the performance variation when using online github runners was too big.  I assume from this that the current data (ie, in [docs/performance](https://github.com/PixarAnimationStudios/OpenUSD/tree/dev/docs/performance), and displayed on [ref_performance_metrics.html](https://openusd.org/release/ref_performance_metrics.html)) was gathered from local machines at Pixar?
- Do you have any of the github actions code used the setup / run the performance tests on github, or information/code on the exact runners or runner requirements used in those tests?  And/or data or logs from those runs?


--------------------------------------------------------------------------------

Specs for current measurements:
https://openusd.org/release/ref_performance_metrics.html#what-environment-is-used

Linux
OS: CentOS Linux 7
CPU: AMD EPYC 7763 64-Core Processor, 2450 Mhz
CPU Utilization: 31 Core(s), 31 Logical Processor(s) (no hyperthreading)
RAM: 117GB
GPU: NVIDIA RTXA6000-24Q

macOS
OS: macOS 14.3
CPU: Apple M2 Ultra (20 Core)
RAM: 192GB
GPU: Apple M2 Ultra GPU (76 Core)

Windows
OS: Microsoft Windows 11 Enterprise
CPU: AMD EPYC 7763 64-Core Processor, 2450 Mhz
CPU Utilization: 31 Core(s), 31 Logical Processor(s) (no hyperthreading)
RAM: 128GB
GPU: NVIDIA RTXA6000-24Q


--------------------------------------------------------------------------------

Requested specs:
https://docs.google.com/document/d/1negkIDb4P2AB5kOOBPs6YLSKUCqCaksUkRJgXJzlxD4/edit?tab=t.0#heading=h.u0bznjlb9ctp

Architectures
Linux / Windows : AMD EPYC 7763 64-Core Processor (or similar)
macOS: Apple M2 Ultra (20 Core) (or newer)

Platform/OS
Linux: el9 or similar
Windows: Windows 11 Professional
macOS: macOS 14.3

GPU cores:
Linux / Windows: NVIDIA RTX A6000 or similar
Memory size: 64GB or similar
Storage size: 500G - 1T SSD


--------------------------------------------------------------------------------


NVIDIA GPU Instances:

Amazon ECS:
https://aws.amazon.com/ec2/instance-types/#Accelerated_Computing

(non-GPUs)
- trn1 / trn1n (AWS Trainium)
- dl1 (Intel/Habana Labs Gaudi)
- vt1 (Xilinx U30)
- inf1 (AWS Inferentia)
- dl2q (Qualcomm AI 100)
- inf2 (AWS Inferentia2)
- f1 (Xilinx Virtex UltaScale+ VU9P FPGA)
- f2 (AMD Virtex UltraScale+ HBM VU47P FPGA)

(GPUs - ruled out)
- g3/g3s (Nvidia M60)
    - GPU too old
- G4dn (Nvidia T4)
    - GPU specs too low
- G4ad (AMD Radeon Pro V520)
- G5g (Nvidia T4G? Special amazon inference-optimized version of T4, runs on ARM)
  - ARM cpu, not valid
- G6e (Nvidia L40S, 48GB)
    - too pricey
- P2 (Nvidia K80)
    - GPU too old
- P3 (Nvidia V100, 16GB)
    - GPU too old
- P3dn (Nvidia V100, 32GB)
    - GPU too old
- P4d (Nvidia A100, 40GB)
    - only available in 96 CPU configs
- P4de (Nvidia A100, 80GB)
    - only available in 96 CPU configs
- P5 (Nvidia H100, 80GB)
    - only in 16 or 192 CPU configs
- P5en/P5e (Nvidia H200, 141GB)
    - only in 192 CPU configs
- P6-b200 (Nvidia B200, 179 GB? 2x 96 = 192GB?)
    - only available with 192 CPUs

(Contenders)

- G5 (Nvidia A10G, 24GB)
  - AMD EPYC 7R32 (48 cores)
    - https://www.cpubenchmark.net/cpu.php?cpu=AMD+EPYC+7R32&id=3894
    - Zen2 - older than 7763, Zen3
    - 2.8 GHz / 3.3 GHz
  - Sizes:
    - g5.8xlarge: 32 CPU, 128GB (1 GPU)
- G6/Gr6/Gr6f/G6f (Nvidia L4, 24GB)
  - AMD EPYC 7R13 48-Core CPU (Zen3)
  - Sizes:
    - g6.8xlarge	32	128	1
    - g6.16xlarge:	64 CPU	256	GB, 1 GPU
    - g6.12xlarge:	48 CPU	192	GB, 4 GPU


Github Runners:
- https://docs.github.com/en/actions/reference/runners/larger-runners#specifications-for-gpu-larger-runners
  - "GPU larger runners"
    - 4 CPU (?), 28 GB RAM
    - 1 Tesla T4, 16GB VRAM

Google Cloud:
https://cloud.google.com/compute/docs/gpus

- A2 Standard (nvidia-a100-40gb)
  - Intel Xeon Platinum 8273CL (28 cores)


macstadium.com:
-----------
https://www.macstadium.com/pricing
Las Vegas / Atlanta

- M4.L Mac Mini
    - M4 Pro 12 Core
    - 48 GB
    - 1 TB SSD
    - $299
- M4.XL Mac mini
    - M4 Pro 14 Core
    - 64 GB
    - 2 TB SSD
    - $399
- S2.M Mac Studio
    - M2 Ultra 24 Core
    - 64 GB
    - 2TB SSD
    - $369/mo

macminivault.com:
-----------------
https://www.macminivault.com/dedicated-mac-mini/
Milwaukee, Wisconsin

- M4 Pro 1TB
    - 14-Core M4 Pro Chip
    - 64GB Unified Memory
    - 1TB SSD
    - 20-core GPU
    - $254.99/mo

--------------------------------------------------------------------------------

Full response from LF re pricing estimate for cloud bare-metal:

    "For the OSX instances I would recommend that we utilize MacStadium's Bare Metal instances: https://www.macstadium.com/pricing

    The S2.M Mac Studio looks to be the closest machine that meets the requirements:

    {{M2 Ultra 24 Core

    64 GB

    2 TB SSD}}

    Those are 369$ a month.

    The linux and windows machines are considerably more expensive, given the CPU/GPU requirements:

    AWS has g6e.16xlarge is a close approximation: 64 vCPUs, 48GB of VRAM, NVIDIA L40S. Depending on the length of the commitment.

    If we go with a 1 year commitment, it works out to roughly 3252$/mo for linux and 5401$/mo for windows. 3 year commitments cut that down by a fair amount:

    2079/mo (linux/3yr reserved instance)

    4028/mo (windows/3yr reserved instance)

    There could potentially be considerable savings if we had some more insight as to the nature of the workloads on these machines. If we can structure the builds such that they only use on demand resources, then we'll end up with significantly smaller bills, and it will be easier to demand dedicated tenancy (which ends up about 8x 'ing the prices).

    If we have flexibility in the number of vCPU's requirements or GPU requirements (i.e. I just went with 48GB of VRAM as the equivalent, but perhaps that's not the important part, and there might be other machines with 24GB that are acceptable etc)."

--------------------------------------------------------------------------------

My Notes:

- Assuming we can run performance metrics on datacenter cards (ie, non-GPU) - need to confirm


- (Note - I'm from NVIDIA, but I was choosing NVIDIA refs because that is what was used in Pixar ref)



--------------------------------------------------------------------------------

GPUs


Ref
---

- RTX A6000
    - Memory:Size (GiB): 48
    - Launch: Oct 5, 2020
    - Ampere
    - GA102-875-A1
    - CUDA 8.6
    - Core clock (MHz): 1410
    - Boost clock (MHz): 1800
    - Tensor compute (FP16) (sparse) TFLOPS: 309.7
    - Processing power - Single-precision TFLOPS: 38.71
    - Similar GeForce Generation: GeForce RTX 3090

    - RTXA6000-24Q is a "virtual GPU" spec - half of an RTXA6000
        - https://docs.nvidia.com/vgpu/13.0/grid-vgpu-user-guide/index.html

    - Processing power - Half-precision TFLOPS: 38.71
    - PCIe 4.0 x16
    - Memory clock (MHz): 2000 (16000)
    - Core config: 10752:336 :112:84:336
    - Cache:L1/SM (KiB): 128
    - Cache:L2 (MiB): 6


Same Gen (lower spec)
---------------------

- A10
    - Memory:Size (GiB): 24
    - Launch: April 12, 2021
    - Ampere (G=)
    - GA102-890-A1
    - CUDA 8.6
    - Core clock (MHz): 885
    - Boost clock (MHz): 1695
    - Half precision tensor core FP32 Accumulate TFLOPS: 125.0
    - Single precision (MAD or FMA) TFLOPS: 31.24
    - Similar GeForce Generation: GeForce RTX 3090

    - Half mem, lower clock, ~40% tensor-FP16, ~80% FP32
- A100
    - Memory:Size (GiB): 40 or 80
    - Launch: May 14, 2020
    - Ampere (G=)
    - GA100-883AA-A1
    - CUDA 8.0
    - Core clock (MHz): 765
    - Boost clock (MHz): 1410
    - Half precision tensor core FP32 Accumulate TFLOPS: 312.0
    - Single precision (MAD or FMA) TFLOPS: 19.5
    - Similar GeForce Generation: GeForce RTX 3090

    - ~similar tensor-FP16, ~50% FP32


Older
-----

- T4 (Amazon g4dn)
    - Memory:Size (GiB): 16
    - Launch: September 12, 2018
    - Turing (G-1)
    - TU104-895-A1
    - CUDA compute: 7.5
    - Core clock (MHz): 585
    - Boost clock (MHz): 1590
    - Half precision tensor core FP32 Accumulate TFLOPS: 64.8
    - Single precision (MAD or FMA) TFLOPS: 8.1
    - Similar GeForce Generation: GeForce RTX 2080 Ti/2080 Super/ Titan RTX

    - 1/3 mem, ~20% tensor-FP16, ~20% FP32

- M60 (Amazon g3)
    - Memory:Size (GiB): 8  (2x??)
    - Maxwell (G-3)
    - 2× GM204-895-A1
    - Launch: August 30, 2015

- K80 (Amazon p2)
    - Kepler (G-4)

- V100 (Amazon p3dn)
    - Memory:Size (GiB): 16 or 32
    - Launch: June 21, 2017
    - Volta (G-1.5)

Newer
-----

- L4 (Amazon g6)
    - Memory:Size (GiB): 24
    - Launch: March 21, 2023
    - Ada Lovelace
    - AD104
    - CUDA 8.9
    - Core clock (MHz): 795
    - Boost clock (MHz): 2040
    - Half precision tensor core FP32 Accumulate TFLOPS: 121.0
    - Single precision (MAD or FMA) TFLOPS: 30.3
    - Similar GeForce Generation: GeForce RTX 3090

    - Half mem, ~40% tensor-FP16, ~80% FP32



--------------------------------------------------------------------------------

CPUs

Ref
---

- AMD EPYC 7763 64-Core Processor, 2450 Mhz
    They were only using 31 cores (1/2) - also half mem, 64/128
    Zen3



32 CPU options:


g5g.8xlarge
$1.372
32
64 GiB
EBS Only
Up to 12 Gigabit

g6.8xlarge
$2.0144
32
128 GiB
2 x 450 GB NVMe SSD
25 Gigabit

g5.8xlarge
$2.448
32
128 GiB
1 x 900 GB NVMe SSD
25 Gigabit

g4dn.8xlarge
$2.176
32
128 GiB
900 GB NVMe SSD
50 Gigabit

gr6.8xlarge
$2.4464
32
256 GiB
2 x 450 GB NVMe SSD
25 Gigabit

g6e.8xlarge
$4.52856
32
256 GiB
1 x 900 GB NVMe SSD
25 Gigabit

g3.8xlarge
$2.28
32
244 GiB
EBS Only
10 Gigabit

p3.8xlarge
$12.24
32
244 GiB
EBS Only
10 Gigabit


--------------------------------------------------------------------------------

Usage stats:
./query_github_pr_pushes.py --start 2025-01-01 --token 'xxx...' PixarAnimationStudios OpenUSD

shows 507 triggering events

~507 hours

Assuming that's over 8 months (2/3 of a year), that's

507 * 3/2 = 760.5 hrs/ year

--------------------------------------------------------------------------------
Pricing calculations, assuming usage growth + inflation (priced per-hour)

If we assume 10% growth/year, for first 5 years, that's

sum([760.5 * 1.1**x for x in range(1, 6)])

= sum(760.5 * [1.1, 1.21, 1.331, 1.4641, 1.61051])

= sum([836.55, 920.205, 1012.2255, 1113.44805, 1224.792855])

= 5107.221405 hours over 5 years

= 1021.444281 hours/year, on average

If we assume PH is current price/hour, and PH increases at 3%/year, then cost over 5 years is:

sum([760.5 * 1.1**x * PH*(1.03)**x for x in range(1, 6)])

= PH * sum([760 * 1.1**x *(1.03)**x for x in range(1, 6)])

= PH * 5613.35131709264

5613.35131709264 / 5 = 1122.670263418528 averaged current-price-hours per-year

--------------------------------------------------------------------------------
Pricing calculations, assuming inflation (priced per-month)

If we assume PM is current price/month, and PM increses at 3%/year, then cost over 5 years is:

sum([12 * PM * (1.03)**x for x in range(1, 6)])

= 12 * PM * sum([1.1, 1.21, 1.331, 1.4641, 1.61051])

= 12 * PM * 6.71561

= PM * 80.58732

--------------------------------------------------------------------------------

Co-location / rack rental notes:

- https://www.colocationamerica.com/colocation/1u-colocation
    - LA area
    - $75/month for 1U
    - $99/month for 2U
    - $199/month for 4U

- https://www.datacentermap.com/

- https://www.lightspeedhosting.com/services/colocation
    - $99/month for 1U
    - Ohio

Rough estimate: $200-$500 /month for colocation

Assuming $200/month

5 year cost, with inflation

= 200 * 80.58732


--------------------------------------------------------------------------------

Pricing Estimate - Buy hardware, Pixar managed

Windows server cost, 5 years:                  $16,500
Linux server cost, 5 years:                    $16,500
Mac server cost, 5 years:                       $1,400
Pixar provided maintenance / hosting: (free to AOUSD?)

Total: $34,400 over 5 years


--------------------------------------------------------------------------------

Pricing Estimate - Cloud bare-metal, 100% usage (original from LF)


- AWS US West (Oregon), g6e.16xlarge
    - 64 vCPU, 512GB mem, 35 gigabit network
    - AMD EPYC 7R13 48-Core CPU (64 cores = 1 1/3 CPUs)
    - 1x NVIDIA L40S GPU (48GB)
    - Windows, 1year committment: $5401/mo
    - Linux, 1year committment: $3252/mo

    - I'm assuming 1 year committment, since we don't want to commit to anything for 3 years right now... but here are
      3-year values for ref:
        - Windows, 3year committment: $4028/mo
        - Linux, 3year committment: $2079/mo

- macstadium.com
    - S2.M Mac Studio
        - M2 Ultra 24 Core
        - 64 GB
        - 2 TB SSD
        - $369/mo

Cloud Windows cost, 5 years:  5401 * 80.58732 = $435,252.12
Cloud Linux cost, 5 years: 3252 * 80.58732    = $262,069.96
Cloud Mac cost, 5 years: 369 * 80.58732       =  $29,736.72

Total: $727,058.80 over 5 years

--------------------------------------------------------------------------------
Pricing Estimate - Cloud on-demand

Cloud instances:

- AWS US West (Oregon), g6.8xlarge
    - 32 vCPU, 128GB mem, 2x450 GBNVMe SSD, 25 gigabit network
    - AMD EPYC 7R13 48-Core CPU (32 cores - 2/3)
    - 1x NVIDIA L4 GPU
    - https://aws.amazon.com/ec2/pricing/on-demand/
    - Windows: $3.4864 / hr
    - Linux: $2.0144 / hr

- macstadium.com
    - M4.L Mac Mini
        - M4 Pro 12 Core
        - 48 GB
        - 1 TB SSD
        - $299

Cloud Windows cost, 5 years: 3.4864 * 5613.35131709264 = $19,570.39
Cloud Linux cost, 5 years: 2.0144 * 5613.35131709264   = $11,307.53
Cloud Mac cost, 5 years: 299 * 80.58732                = $24,095.61

Total: $54,973.53 over 5 years

--------------------------------------------------------------------------------

Pricing Estimate - Buy hardware, AOUSD managed

Windows server cost, 5 years:               $16,500.00
Linux server cost, 5 years:                 $16,500.00
Mac server cost, 5 years:                    $1,400.00
LA datacenter colocation: 200 * 80.58732 =  $16,117.46

Total: $50,517 over 5 years

