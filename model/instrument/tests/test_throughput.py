"""The money guard: a pool that holds GPUs without using them is stopped early."""

from model.instrument.throughput import CHECK_AFTER, observed_concurrency, stop_message

BOOT = 120.0  # container start-up, charged against elapsed but against no run


def parallel(n_landed, run_seconds, containers):
    """Elapsed when `n_landed` results are in from a pool that really is parallel."""
    waves = (n_landed + containers - 1) // containers
    return [run_seconds] * n_landed, BOOT + waves * run_seconds


def serial(n_landed, run_seconds):
    return [run_seconds] * n_landed, BOOT + n_landed * run_seconds


def test_a_healthy_pool_is_never_stopped_at_any_point_of_the_first_wave():
    # The first results land together, so concurrency reads low until the whole
    # wave is in. The guard has to survive that without a false alarm.
    for landed in range(CHECK_AFTER, 9):
        seconds, elapsed = parallel(landed, 2438.0, 8)
        assert stop_message(seconds, elapsed, "L4", 8) == ""


def test_a_serial_pool_is_stopped_as_soon_as_it_can_be_told_apart():
    seconds, elapsed = serial(CHECK_AFTER, 520.0)
    message = stop_message(seconds, elapsed, "L40S", 8)
    assert message.startswith("STOPPING")
    assert "L40S" in message and "nvidia-smi" in message
    # It says what is being wasted, in the terms the ledger is kept in.
    assert "holding a GPU" in message and "paid for" in message


def test_the_real_numbers_from_the_incident_would_have_been_caught():
    # Six runs of 590 s landed in 65 minutes on a pool of eight.
    assert stop_message([590.0] * 6, 65 * 60, "L40S", 8).startswith("STOPPING")
    # And the L4 sweep that worked would not have been.
    assert stop_message([178.0] * 8, BOOT + 178.0, "L4", 8) == ""


def test_nothing_is_judged_before_there_is_enough_to_judge():
    seconds, elapsed = serial(CHECK_AFTER - 1, 520.0)
    assert stop_message(seconds, elapsed, "L40S", 8) == ""


def test_a_pool_of_one_is_serial_on_purpose_and_is_left_alone():
    seconds, elapsed = serial(5, 520.0)
    assert stop_message(seconds, elapsed, "L4", 1) == ""


def test_half_speed_is_tolerated_and_a_quarter_is_not():
    # Four of eight containers busy: slower than hoped, not worth losing runs over.
    assert stop_message([100.0] * 4, 100.0, "L4", 8) == ""
    # One of eight: stopped.
    assert stop_message([100.0] * 4, 400.0, "L4", 8).startswith("STOPPING")


def test_concurrency_is_the_ratio_of_work_to_wall_clock():
    assert observed_concurrency([100.0, 100.0], 50.0) == 4.0
    assert observed_concurrency([], 50.0) == 0.0
    # An elapsed of zero cannot divide, and is not a reason to raise.
    assert observed_concurrency([100.0], 0.0) == 0.0
    assert stop_message([100.0], 0.0, "L4", 8) == ""


def test_the_shape_that_produced_the_false_alarm_is_documented():
    # 27 runs of about 1,580 s finished in 96 min, which is 7.4 containers
    # busy, but ordered output yielded only three results in that time and the
    # guard read 0.8. The guard is right about what it is given; the caller has
    # to give it arrival times, which is why the map is unordered now.
    assert stop_message([1584.0] * 3, 96 * 60, "L4", 8).startswith("STOPPING")
    # The same work, reported as it lands, passes.
    assert stop_message([1584.0] * 8, 120 + 1584.0, "L4", 8) == ""


def test_a_busy_pool_on_uneven_runs_is_not_mistaken_for_an_idle_one():
    """Twelve short runs land first, eight long ones are all still in flight.

    Twelve landed in 1,128 s on eight containers with 4,460 s of work: the ratio
    says 3.9 busy while every container is running a long job. That fired the
    guard, the client exited, and the app killed the six jobs in flight.
    """
    short = [350.0] * 12 + [700.0]
    assert stop_message(short, 1128.0, "L4", 8) == ""
    # The first wave is still judged: eight results that landed one after
    # another is a serial pool.
    assert stop_message([300.0] * 8, 8 * 300.0 + 60, "L4", 8).startswith("STOPPING")
