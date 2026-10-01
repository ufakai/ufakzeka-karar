"""Is the container pool actually working, or just allocated?

A dispatch that runs one job at a time while holding eight GPUs looks healthy
run by run: every run finishes in the time it should, nothing fails, and the
log fills at a steady rate. Only the throughput gives it away. One sweep cost an
hour of L40S time to exactly that, with `nvidia-smi` inside the containers
reporting 0 percent utilisation and 7 MiB of 46 GB in use.

So concurrency is measured from the results themselves and checked early,
rather than read off the number of containers the scheduler reports.
"""

from __future__ import annotations

# Judged once this many runs have landed: enough to tell a parallel pool from
# a serial one, cheap enough to lose if the answer is bad.
CHECK_AFTER = 3
# Below this share of the expected concurrency the dispatch is stopped rather
# than left to run.
MIN_SHARE = 0.5


def observed_concurrency(seconds: list[float], elapsed: float) -> float:
    """How many containers were busy on average while the dispatch was open.

    The runs that have landed account for `sum(seconds)` of GPU work and the
    dispatch has been open for `elapsed`, so their ratio is the average number
    of containers doing work. It needs no cooperation from the scheduler and
    no second measurement.

    This is only true when results arrive as they finish. Under Modal's
    default ordering a finished result waits for every earlier input, so the
    arrival times are serialised by the slowest run and this ratio collapses
    towards one however busy the pool is. The caller must map with
    `order_outputs=False`.
    """
    if elapsed <= 0:
        return 0.0
    return sum(seconds) / elapsed


def stop_message(seconds: list[float], elapsed: float, gpu: str, max_containers: int) -> str:
    """Empty while the pool is keeping up, a stop message when it is not.

    In a healthy pool the first results land together, so after n results the
    observed concurrency is close to n until n reaches the pool size. In a
    serial one it sits near 1 however many land. Half of the expected value
    separates the two cleanly and leaves room for container start-up, which
    counts against elapsed but not against any run's seconds.
    """
    if max_containers < 2 or len(seconds) < CHECK_AFTER or elapsed <= 0:
        return ""
    if len(seconds) > max_containers:
        # Judged on the first wave only. The ratio counts the seconds of runs
        # that have landed and nothing of the runs still in flight, so once
        # the pool holds a second wave the measure only falls, and on a batch
        # of short and long runs it falls below the threshold while every
        # container is busy. That stopped a client whose app then killed six
        # running jobs. A pool that never fans out shows in its first
        # wave, which is where this looks.
        return ""
    observed = observed_concurrency(seconds, elapsed)
    expected = min(len(seconds), max_containers)
    if observed >= MIN_SHARE * expected:
        return ""
    idle = (max_containers - observed) / max_containers
    return (
        f"STOPPING: {len(seconds)} runs landed in {elapsed / 60:.1f} min, which is "
        f"{observed:.1f} containers busy on average against {expected} expected of "
        f"{max_containers} on {gpu}. About {idle:.0%} of the pool is holding a GPU and "
        "doing nothing, and that is paid for. Check with "
        "`modal container exec <id> -- nvidia-smi` before launching again, and use a GPU "
        "the account has capacity for. Finished runs are kept, so resuming pays "
        "for none of them twice."
    )
