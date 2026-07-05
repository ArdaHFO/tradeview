from datetime import datetime, timezone

from scanner.session import (NY, in_entry_window, is_opening_drive, is_premarket,
                             is_rth, minutes_since_open, minutes_to_close,
                             session_elapsed_fraction, to_ny)


def ny(h, m, day=11):
    return datetime(2026, 6, day, h, m, tzinfo=NY)


def test_rth_boundaries():
    assert not is_rth(ny(9, 29))
    assert is_rth(ny(9, 30))
    assert is_rth(ny(15, 59))
    assert not is_rth(ny(16, 0))


def test_weekend():
    assert not is_rth(ny(10, 0, day=13))   # Saturday 2026-06-13


def test_premarket():
    assert is_premarket(ny(8, 0))
    assert not is_premarket(ny(9, 30))
    assert not is_premarket(ny(3, 0))


def test_minutes_since_open_and_close():
    assert minutes_since_open(ny(10, 30)) == 60.0
    assert minutes_to_close(ny(15, 45)) == 15.0


def test_entry_window_lockout():
    assert in_entry_window(ny(15, 44))
    assert not in_entry_window(ny(15, 45))
    assert not in_entry_window(ny(15, 50))


def test_opening_drive():
    assert is_opening_drive(ny(9, 45))
    assert not is_opening_drive(ny(11, 0))


def test_elapsed_fraction():
    assert session_elapsed_fraction(ny(9, 30)) == 0.0
    assert abs(session_elapsed_fraction(ny(12, 45)) - 0.5) < 1e-9
    assert session_elapsed_fraction(ny(17, 0)) == 1.0


def test_utc_naive_assumed_utc():
    # 14:30 UTC == 10:30 ET in June (EDT)
    naive = datetime(2026, 6, 11, 14, 30)
    assert to_ny(naive).hour == 10
    aware = datetime(2026, 6, 11, 14, 30, tzinfo=timezone.utc)
    assert is_rth(aware)
