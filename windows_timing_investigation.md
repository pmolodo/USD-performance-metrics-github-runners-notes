# Investigation: short successful Windows builds

Investigation date: 2026-10-02

## Finding

The sampled short Windows jobs are genuine successful executions with very
high compiler-cache hit rates. The logs show executed tests, uploaded artifacts,
and ccache hit rates above 99.9%. No evidence of falsely successful builds was
found in the inspected sample. The 8.79-minute p10 is plausible for this cached
workflow, but it is not an estimate of a clean compilation from scratch.

The timing dataset contains 375 distinct successful Windows executions, of
which 105 (28%) finish in under 15 minutes. All 105 occur from June 2026 onward.
The 15-minute boundary is an investigation aid, not a new exclusion threshold.
The timing filters and graphs have not been changed by this investigation.

## Direct log evidence

Five short jobs with available logs were inspected, covering July through
October, including the shortest recorded success and samples near the p10.
A long job on October 1 provides a comparison. All six showed successful
build/test steps, artifact uploads, and passing test summaries.

| Job date and link | Full job minutes | Build step minutes | Compiler-cache hit rate | Main tests passed | Additional flaky tests passed |
|---|---:|---:|---:|---:|---:|
| [2026-07-07](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/28900091921/job/85734564172) | 8.78 | 3.18 | 99.93% | 1,030 | 4 |
| [2026-08-20](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/32317362931/job/96272491532) | 8.80 | 2.98 | 99.97% | 1,032 | 4 |
| [2026-09-09](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/34414273753/job/102675620480) | 8.87 | 3.02 | 99.97% | 1,043 | 4 |
| [2026-09-23, shortest](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/35834606047/job/107095113412) | 6.98 | 2.70 | 99.97% | 1,045 | 4 |
| [2026-10-01, short](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/36880946877/job/110432491227) | 7.88 | 3.23 | 99.97% | 1,058 | 4 |
| [2026-10-01, long](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/36880389663/job/110430638857) | 96.12 | 90.73 | 15.92% | 1,058 | 4 |

The two October 1 jobs are especially informative: they execute the same
number of tests, but the short job records 2,889 cache hits out of 2,890
cacheable compiler calls, whereas the long job records 460 hits and 2,430
misses. The short job spends about three minutes in the build step; the long
one spends about 91 minutes. These are different commits, not a controlled
performance experiment, but the cache statistics directly explain why most
compilation work is avoided in the short execution.

Both jobs report a successfully restored cache. Restoring a cache does not
imply that its entries match most of the current build's compiler calls; the
hit/miss counts are the relevant evidence.

The [workflow at the short October job's commit](https://github.com/PixarAnimationStudios/OpenUSD/blob/240e3ddef8d79a5800490c85ca4a7741397f8516/.github/workflows/buildusd.yml#L260)
configures `hendrikmuhs/ccache-action`, a Windows-specific cache key, and a
3 GB maximum cache. The observed test summaries and cache statistics establish
more than the job's success flag alone.

## Artifact-size cross-check

GitHub's [workflow-run artifacts API](https://docs.github.com/en/rest/actions/artifacts#list-workflow-run-artifacts)
returns `size_in_bytes`. Summing that field for the expected output names gives
their total archive size without downloading the artifacts. The API associates
artifacts with a workflow run, not directly with a job, so this comparison
selects `usd-win64` and checks its creation time against the job's execution.
Each available sample has exactly one matching artifact, created during its
Windows job. Multiple attempts or identically named outputs require additional
attribution checks rather than summing the entire workflow's outputs.

| Windows job | Minutes | `usd-win64` archive MiB |
|---|---:|---:|
| July 7 | 8.78 | 68.546 |
| August 20 | 8.80 | 68.669 |
| September 9 | 8.87 | 68.572 |
| September 23 | 6.98 | 64.257 |
| October 1, short | 7.88 | 64.290 |
| October 1, long | 96.12 | 64.291 |

The short and long October 1 jobs produced 67,413,195 and 67,413,504 bytes
respectively: a difference of only 309 bytes. Combined with passing tests and
the cache statistics, this strongly supports valid cached builds rather than
tiny or missing outputs caused by a broken build.

Artifact size is a useful sanity signal, not proof of correctness. Flag missing
expected outputs or unusually small archives relative to comparable builds
of the same configuration. Account for changes in packaging, debug symbols,
architectures, and compression. Treat artifacts missing after their retention
period as unknown, not failed. The June 1 sample has no retained artifacts.
No size-based exclusion rule has been applied to the timing dataset.

Exact sizes and timestamps are in
[windows_timing_artifact_samples.csv](windows_timing_artifact_samples.csv).

## Coverage limits

The first short cached execution is
[June 1, 2026](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/26787765376/job/78967472353),
at 8.37 minutes. Its live metadata still says success, but step details are
absent and its log download returns HTTP 410. It cannot be log-verified now.
This investigation does not prove every historical short job was valid.

Persistent Windows compiler caching was introduced on **2026-05-28** (UTC
commit timestamp 19:42:25), in
[528289b55f035f660f852c69fe930ebe50dac6fc](https://github.com/PixarAnimationStudios/OpenUSD/commit/528289b55f035f660f852c69fe930ebe50dac6fc).
The patch adds `ccache-action` and `--compiler-cache`, replaces the installed-
dependency cache, and switches from the Visual Studio generator to Ninja.
The Windows timing graphs mark this date. The simultaneous generator change
means the before/after difference is not a controlled measure of caching alone.
The first retained sub-15-minute success is June 1; individual branches and
in-flight workflows need not adopt the change on the commit date.

| Month | Successful Windows executions | Under 15 minutes |
|---|---:|---:|
| 2025-07 through 2026-05 | 219 | 0 |
| 2026-06 | 19 | 13 |
| 2026-07 | 36 | 27 |
| 2026-08 | 35 | 25 |
| 2026-09 | 64 | 39 |
| 2026-10, partial | 2 | 1 |

The selected available logs support the cache explanation for the low p10.
Rejecting jobs solely for being shorter than 15 minutes would remove many
valid, successfully tested builds and bias the estimate upward.

## Implications for cost projections

Keep the successful-job p10 in the observational report. For forecasting,
separate high-cache-hit builds from low-cache-hit builds and model the expected
mix. A new configuration matrix can reduce cache reuse through different
compiler options, architectures, and cache capacity. Do not assume that the
seven-to-nine-minute path will apply to all 16 configurations.

The 96-minute sample is a low-hit build, not a fully cold-cache benchmark.
Neither it nor the pooled 70.28-minute historical mean alone establishes a
representative duration for the proposed runner hardware. A controlled warm-
and cold-cache benchmark of the actual configurations would provide stronger
budget inputs. Failed attempts still need a separate compute allowance.

## Why Linux and macOS do not show the same short-build mode

The [sampled workflow](https://github.com/PixarAnimationStudios/OpenUSD/blob/240e3ddef8d79a5800490c85ca4a7741397f8516/.github/workflows/buildusd.yml)
uses different persistent caches:

| Platform | Cache restored between jobs | Consequence |
|---|---|---|
| Linux | `USDinst`, keyed by OS, Python version, and build-script hash | Installed dependencies can be reused; USD build intermediates are not restored |
| macOS | `USDinst`, with the same key structure | Installed dependencies can be reused; USD build intermediates are not restored |
| Windows | Explicit `ccache-action`, Windows cache key, 3 GB limit | Matching compiler outputs can be reused, producing very short high-hit builds |

Linux and macOS do pass `-DPXR_ENABLE_COMPILER_CACHE=ON`. That enables compiler
cache support in the build configuration; it does not itself preserve a cache
across jobs. Their workflow does not restore/save a compiler-cache directory
or the `USDgen/build` tree. GitHub-hosted jobs use
[fresh runner instances](https://docs.github.com/en/actions/how-tos/write-workflows/choose-where-workflows-run/choose-the-runner-for-a-job).
Consequently, they lack the persistent warm compiler cache that explains the
Windows high-hit path. The flag alone does not establish whether a cache tool
was found or how many hits it achieved; the Linux/macOS logs inspected here
do not report compiler-cache hit statistics.

Restoring `USDinst` does not skip the USD build. The
[build script](https://github.com/PixarAnimationStudios/OpenUSD/blob/240e3ddef8d79a5800490c85ca4a7741397f8516/build_scripts/build_usd.py#L2968)
installs missing dependencies and then invokes the USD installer unconditionally.
Object files and incremental build state live in the separate, uncached build
directory. Ccache can avoid repeated compilation only when matching cached
compiler results are available; see the [ccache manual](https://ccache.dev/manual/latest.html).

Log checks on October 1:

- [Linux](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/36880389663/job/110430638979):
  installation-cache hit, no dependencies to build, but 74.07 minutes in the
  USD build step and 78.57 minutes for the full job.
- [macOS](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/36880946877/job/110432491174):
  installation-cache miss; oneTBB, OpenSubdiv, and USD built. The build step
  took 67.23 minutes and the full job took 70.67 minutes.

The cache design explains why Windows has a pronounced seven-to-nine-minute
mode while the other platforms still spend substantial time building USD.
Dependency-cache misses, runner/toolchain differences, and test scope can
contribute additional variation. These observational checks do not quantify
the speedup a persistent compiler cache would produce on Linux or macOS.
In particular, the present platform timings should not be treated as an
apples-to-apples comparison of hardware performance under identical caching.

## August 2026 Windows p90 drop

The drop reflects a much shorter slow-build tail, not a comparable speedup
of the typical warm-cache build. The precise underlying cause is unresolved.

| Successful Windows jobs | July 2026 | August 2026 |
|---|---:|---:|
| Sample count | 36 | 35 |
| Full-job median (minutes) | 9.30 | 9.53 |
| Full-job p90 (minutes) | 159.28 | 76.65 |
| Build-step p90 (minutes) | 154.03 | 70.47 |
| Maximum full-job duration (minutes) | 169.85 | 94.10 |
| Jobs longer than 120 minutes | 6 | 0 |

Better cache-hit rates alone do not explain this. For example:

- [July 8 job](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/28955606440/job/85913733551):
  169.85 minutes, 21.62% compiler-cache hits, Windows image
  `20260628.224.1`, Visual Studio developer prompt version `17.14.35`.
- [August 20 job](https://github.com/PixarAnimationStudios/OpenUSD/actions/runs/32406927365/job/96548314927):
  94.10 minutes, 17.26% compiler-cache hits, Windows image
  `20260802.262.1`, Visual Studio developer prompt version `17.14.37`.

Both used ccache 4.13, Ninja, four build workers, Release configuration,
Python 3.9.13, precompiled headers disabled, and compiler caching enabled.
They made about 2,870-2,880 cacheable compiler calls and passed more than
1,030 tests. These are substantial successful builds, not obvious early exits.

The `dev` history for `.github/workflows/buildusd.yml` contains no August
commits. July changes concern test-shell handling and Linux test exclusions;
inspection of July-August `build_scripts/build_usd.py` changes did not identify
an obvious Windows speed optimization. Windows caching was enabled on May 28,
not in August.

A runner-image/toolchain change is a plausible explanation worth testing, but
the logs do not establish causality. The sampled slow July jobs used Visual
Studio 17.14.35; August samples used 17.14.37 or 17.14.39. The
[August runner-image release](https://github.com/actions/runner-images/releases/tag/win22%2F20260802.262)
documents an environment refresh, but the
[Visual Studio release notes](https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-notes)
do not identify a C++ build-performance fix for 17.14.36 or 17.14.37.
Different source changes, the particular files missing from cache, and runner
performance remain confounders. A cache-hit percentage counts calls, not their
compilation cost, so equal percentages do not mean equal work.

The small monthly samples also make p90 sensitive to a handful of jobs.
July's interpolated p90 falls between its fifth- and fourth-slowest jobs
(158.75 and 159.82 minutes); August's falls between 62.28 and 86.23 minutes.
The data support a reduction in observed slow-build durations, but not a
specific proven optimization or a guaranteed sustained speedup.

## Reproduction

From the analysis submodule:

```sh
uv run python investigate_windows_timings.py
uv run python inspect_native_cache_logs.py
uv run python find_windows_cache_change.py
uv run python investigate_windows_p90.py
```

The script uses `requests` for GitHub reads and the existing `gh` authentication
token without displaying it. Job metadata, artifact metadata, public logs, and workflow files
are cached in `.cache/windows_timing_investigation/`. Cached material is local
and is not added to git. Signed log-download URLs are not recorded.
The API behavior is documented in
[GitHub's job-log endpoint](https://docs.github.com/en/rest/actions/workflow-jobs#download-job-logs-for-a-workflow-run).
The p90 investigation caches sampled logs and workflow/build-script commit
history separately in `.cache/windows_p90_investigation/`.

This is a diagnostic report; no exclusion rule, timing statistic, or cost
projection was changed.
