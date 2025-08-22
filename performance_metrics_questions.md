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

- G4dn (T4)
- G5 (A10G, 24GB)
  - AMD EPYC 7R32 (48 cores)
    - https://www.cpubenchmark.net/cpu.php?cpu=AMD+EPYC+7R32&id=3894
    - Zen2 - older than 7763, Zen3
    - 2.8 GHz / 3.3 GHz
  - Sizes:
    - g5.4xlarge: 16 CPU, 64GB (1 GPU)
    - g5.8xlarge: 32 CPU, 128GB (1 GPU)
    - g5.16xlarge: 64 CPU, 256GB (1 GPU)
    - g5.12xlarge: 48 CPU, 192GB (4 GPU)
    - g5.48xlarge: 192 CPU, 768GB (8 GPU)
- G5g (T4G? Special amazon inference-optimized version of T4, runs on ARM)
  - ARM cpu, not valid
- G6 (L4, 24GB)
  - AMD EPYC 7R13 48-Core CPU (Zen3) 
  - Sizes:
    - g6.4xlarge	16	64	1
    - g6.8xlarge	32	128	1
    - g6.16xlarge:	64 CPU	256	GB, 1 GPU
    - g6.12xlarge:	48 CPU	192	GB, 4 GPU
- G6g (L40S, 48GB)


Github Runners:
- https://docs.github.com/en/actions/reference/runners/larger-runners#specifications-for-gpu-larger-runners
  - "GPU larger runners"
    - 4 CPU (?), 28 GB RAM
    - 1 Tesla T4, 16GB VRAM

Google Cloud:
https://cloud.google.com/compute/docs/gpus

- A2 Standard (nvidia-a100-40gb)
  - Intel Xeon Platinum 8273CL (28 cores)
Azure:


--------------------------------------------------------------------------------

My Notes:

- Assuming we can run performance metrics on datacenter cards (ie, non-GPU) - need to confirm


- (Note - I'm from NVIDIA, but I was choosing NVIDIA refs because that is what was used in Pixar ref)



- RTXA6000-24Q is a "virtual GPU" spec - half of an RTXA6000
    - https://docs.nvidia.com/vgpu/13.0/grid-vgpu-user-guide/index.html



- RTXA6000
    - GA102-875-A1
    - PCIe 4.0 x16
    - Core clock (MHz): 1410
    - Boost clock (MHz): 1800
    - Memory clock (MHz): 2000 (16000)
    - Core config: 10752:336 :112:84:336
    - Cache:L1/SM (KiB): 128
    - Cache:L2 (MiB): 6
    - Memory:Size (GiB): 48