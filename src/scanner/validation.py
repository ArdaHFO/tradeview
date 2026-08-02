"""Statistical validation of backtest results: is the edge real, or is it noise?

A profit factor read off a single backtest is a point estimate from one sample
path. That number alone cannot tell you whether a strategy is worth trading.
This module answers the questions that actually decide it:

  * bootstrap    -- resample the trade list with replacement to put a confidence
                    interval around expectancy and profit factor. If the
                    interval straddles break-even, the edge is NOT established,
                    however good the point estimate looks.
  * monte carlo  -- reshuffle trade ORDER (the realized sequence is arbitrary;
                    only the distribution of outcomes carries information) to
                    see the range of equity paths and drawdowns the same edge
                    could plausibly have produced.
  * sample size  -- given the observed mean/stdev of R, how many trades would be
                    needed before the confidence interval clears zero at all.

Both resampling methods are order statistics over the realized trades, so they
inherit every bias in the backtest itself -- notably the synthetic order flow
derived from bar shape. They establish whether the measured edge is
distinguishable from luck; they say nothing about whether the backtest was
faithful to live markets. Treat a passing verdict as "worth paper trading",
never as "worth funding".
"""
from __future__ import annotations

import json
import random
import re
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

DEFAULT_ITERATIONS = 10_000
DEFAULT_SEED = 42
Z_95 = 1.959964            # two-sided 95% normal quantile


@dataclass(frozen=True)
class Trade:
    """One closed backtest trade, flattened for statistics."""
    day: str                # YYYY-MM-DD
    time: str               # HH:MM
    symbol: str
    setup: str
    side: str
    entry: float
    exit: float
    exit_reason: str
    r_multiple: float
    pnl: float


# --- ingest ---------------------------------------------------------------

# Matches a trade line emitted by backtest.format_report(), e.g.
#   15:00 AKTX   VWAP_REVERSION SHORT in    18.88 out    19.32 (stop  ) R -1.17  PnL $  -58.30
_TRADE_RE = re.compile(
    # Symbol charset allows digits and lowercase: preferred/warrant tickers look
    # like AHTpF, and class shares like BRK.B.
    r"^\s*(?P<time>\d{2}:\d{2})\s+(?P<symbol>[A-Za-z0-9.\-]+)\s+(?P<setup>[A-Z_]+)\s+"
    r"(?P<side>LONG|SHORT)\s+in\s+(?P<entry>[\d.]+)\s+out\s+(?P<exit>[\d.]+)\s+"
    r"\((?P<reason>[a-z]+)\s*\)\s+R\s+(?P<r>[+-][\d.]+)\s+"
    r"PnL\s+\$\s*(?P<pnl>[+-][\d,.]+)\s*$")
_DAY_RE = re.compile(r"^=== BACKTEST (?P<day>\d{4}-\d{2}-\d{2})")


def trades_from_log(path: str | Path) -> list[Trade]:
    """Parse trades out of a saved `python main.py backtest` console log.

    Lets us validate runs that were captured before JSON export existed,
    instead of paying hours of throttled API time to reproduce them.
    """
    trades: list[Trade] = []
    day = ""
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        d = _DAY_RE.match(line)
        if d:
            day = d.group("day")
            continue
        m = _TRADE_RE.match(line)
        if not m:
            continue
        trades.append(Trade(
            day=day, time=m.group("time"), symbol=m.group("symbol"),
            setup=m.group("setup"), side=m.group("side"),
            entry=float(m.group("entry")), exit=float(m.group("exit")),
            exit_reason=m.group("reason"), r_multiple=float(m.group("r")),
            pnl=float(m.group("pnl").replace(",", ""))))
    return trades


def trades_from_reports(reports) -> list[Trade]:
    """Convert live `BacktestReport` objects (backtest.py) into Trade rows."""
    out: list[Trade] = []
    for rep in reports:
        for r in rep.results:
            s = r.signal
            out.append(Trade(
                day=rep.trading_day.isoformat(), time=s.ts.strftime("%H:%M"),
                symbol=s.symbol, setup=s.setup.value, side=s.side.value,
                entry=s.entry, exit=r.exit_price, exit_reason=r.exit_reason,
                r_multiple=r.r_multiple, pnl=r.pnl))
    return out


def save_trades(trades: list[Trade], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(t) for t in trades], indent=1),
                          encoding="utf-8")


def load_trades(path: str | Path) -> list[Trade]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Trade(**t) for t in raw]


# --- point metrics --------------------------------------------------------

def profit_factor(rs: list[float]) -> float:
    """PF over R-multiples. inf when there are wins but no losses."""
    wins = sum(r for r in rs if r > 0)
    losses = -sum(r for r in rs if r < 0)
    if losses <= 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def win_rate(rs: list[float]) -> float:
    return sum(1 for r in rs if r > 0) / len(rs) if rs else 0.0


def equity_curve(pnls: list[float], start: float = 0.0) -> list[float]:
    curve, running = [start], start
    for p in pnls:
        running += p
        curve.append(running)
    return curve


def max_drawdown(curve: list[float]) -> float:
    """Largest peak-to-trough decline along an equity curve, in currency."""
    peak, worst = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        worst = max(worst, peak - v)
    return worst


# --- resampling -----------------------------------------------------------

@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def excludes(self, value: float) -> bool:
        return self.lo > value or self.hi < value


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]. Input must be sorted."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


@dataclass
class BootstrapResult:
    n_trades: int
    iterations: int
    expectancy_r: Interval        # mean R per trade
    profit_factor: Interval
    win_rate: Interval
    p_no_edge: float              # share of resamples with mean R <= 0
    trades_needed: int | None     # trades required for a 95% CI to clear zero


def bootstrap(trades: list[Trade], iterations: int = DEFAULT_ITERATIONS,
              seed: int = DEFAULT_SEED, alpha: float = 0.05) -> BootstrapResult:
    """Resample trades with replacement; return CIs for the headline metrics.

    Expectancy (mean R) is the primary statistic: it is well behaved under
    resampling, whereas PF can blow up to infinity on a draw that happens to
    contain no losers.
    """
    rs = [t.r_multiple for t in trades]
    n = len(rs)
    if n < 2:
        raise ValueError("need at least 2 trades to bootstrap")

    means: list[float] = []
    pfs: list[float] = []
    wins: list[float] = []
    rng = random.Random(seed)
    for _ in range(iterations):
        draw = [rs[int(rng.random() * n)] for _ in range(n)]
        means.append(sum(draw) / n)
        pf = profit_factor(draw)
        if pf != float("inf"):          # degenerate all-winner draw: skip for CI
            pfs.append(pf)
        wins.append(win_rate(draw))
    means.sort(), pfs.sort(), wins.sort()

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    observed_mean = sum(rs) / n
    sd = statistics.stdev(rs)
    needed = None
    if observed_mean > 0 and sd > 0:
        # n at which the 95% CI half-width (z*sd/sqrt(n)) shrinks below the mean
        needed = int((Z_95 * sd / observed_mean) ** 2) + 1

    return BootstrapResult(
        n_trades=n, iterations=iterations,
        expectancy_r=Interval(observed_mean, _percentile(means, lo_q),
                              _percentile(means, hi_q)),
        profit_factor=Interval(profit_factor(rs), _percentile(pfs, lo_q),
                               _percentile(pfs, hi_q)),
        win_rate=Interval(win_rate(rs), _percentile(wins, lo_q),
                          _percentile(wins, hi_q)),
        p_no_edge=sum(1 for m in means if m <= 0) / len(means),
        trades_needed=needed)


@dataclass
class MonteCarloResult:
    iterations: int
    start_equity: float
    realized_pnl: float
    realized_max_dd: float
    final_pnl: Interval           # 95% band of end-of-run PnL
    max_dd: Interval              # 95% band of worst drawdown
    shuffle_max_dd: Interval      # DD band from pure reordering (edge held fixed)
    p_losing_run: float           # share of simulated runs ending below break-even
    p_deep_dd: float              # share breaching `dd_limit_pct` of equity
    dd_limit_pct: float


def monte_carlo(trades: list[Trade], start_equity: float,
                iterations: int = DEFAULT_ITERATIONS, seed: int = DEFAULT_SEED,
                dd_limit_pct: float = 10.0,
                alpha: float = 0.05) -> MonteCarloResult:
    """Simulate alternative runs of the same strategy to map path risk.

    Two resamplings, because they answer different questions:

    * *resample with replacement* (primary) -- draw N trades from the observed
      distribution to build a run that could have happened instead. This is what
      makes "how often does a run like this end in the red?" meaningful.
    * *reorder only* -- shuffle the realized trades. The total is invariant under
      reordering, so this says nothing about final PnL, but it isolates how much
      of the realized drawdown was pure sequencing luck.
    """
    pnls = [t.pnl for t in trades]
    n = len(pnls)
    if n < 2:
        raise ValueError("need at least 2 trades for a monte carlo run")
    limit = start_equity * dd_limit_pct / 100.0

    finals: list[float] = []
    dds: list[float] = []
    shuffle_dds: list[float] = []
    rng = random.Random(seed)
    reordered = list(pnls)
    for _ in range(iterations):
        draw = [pnls[int(rng.random() * n)] for _ in range(n)]
        finals.append(sum(draw))
        dds.append(max_drawdown(equity_curve(draw)))
        rng.shuffle(reordered)
        shuffle_dds.append(max_drawdown(equity_curve(reordered)))
    finals.sort(), dds.sort(), shuffle_dds.sort()

    lo_q, hi_q = alpha / 2.0, 1.0 - alpha / 2.0
    realized_dd = max_drawdown(equity_curve(pnls))
    return MonteCarloResult(
        iterations=iterations, start_equity=start_equity,
        realized_pnl=sum(pnls), realized_max_dd=realized_dd,
        final_pnl=Interval(sum(pnls), _percentile(finals, lo_q),
                           _percentile(finals, hi_q)),
        max_dd=Interval(realized_dd, _percentile(dds, lo_q),
                        _percentile(dds, hi_q)),
        shuffle_max_dd=Interval(realized_dd, _percentile(shuffle_dds, lo_q),
                                _percentile(shuffle_dds, hi_q)),
        p_losing_run=sum(1 for f in finals if f <= 0) / len(finals),
        p_deep_dd=sum(1 for d in dds if d >= limit) / len(dds),
        dd_limit_pct=dd_limit_pct)


# --- reporting ------------------------------------------------------------

@dataclass
class Verdict:
    passed: bool
    headline: str
    notes: list[str] = field(default_factory=list)


def verdict(boot: BootstrapResult, mc: MonteCarloResult,
            pf_target: float = 1.3) -> Verdict:
    """Turn the statistics into a go / no-go call with the reasoning attached."""
    notes: list[str] = []
    edge_proven = boot.expectancy_r.lo > 0

    if edge_proven:
        head = "EDGE İSTATİSTİKSEL OLARAK ANLAMLI"
    else:
        head = "EDGE KANITLANMADI — sonuç gürültüden ayırt edilemiyor"
        notes.append(
            f"Beklenti (ortalama R) %95 güven aralığı [{boot.expectancy_r.lo:+.3f}, "
            f"{boot.expectancy_r.hi:+.3f}] sıfırı içeriyor: aynı sonuç şansla da çıkabilirdi.")
        if boot.trades_needed:
            notes.append(
                f"Bu etki büyüklüğünü kanıtlamak için ~{boot.trades_needed:,} işlem "
                f"gerekir (elde {boot.n_trades}).")
        else:
            notes.append("Gözlenen beklenti pozitif değil: örneklem büyütmek yetmez, "
                         "stratejinin kendisi değişmeli.")

    if boot.profit_factor.lo < 1.0 <= boot.profit_factor.point:
        notes.append(
            f"PF nokta tahmini {boot.profit_factor.point:.2f} ama alt sınır "
            f"{boot.profit_factor.lo:.2f} — başabaşın altı hâlâ makul bir sonuç.")
    if boot.profit_factor.hi < pf_target:
        notes.append(
            f"PF üst sınırı {boot.profit_factor.hi:.2f}, hedef {pf_target:.2f}'in altında: "
            "bu veriyle hedefe ulaşmak istatistiksel olarak mümkün görünmüyor.")
    if mc.p_losing_run > 0.05:
        notes.append(
            f"Aynı dağılımdan üretilen koşuların %{mc.p_losing_run*100:.0f}'i "
            "zararla bitiyor.")
    if mc.p_deep_dd > 0.05:
        notes.append(
            f"Koşuların %{mc.p_deep_dd*100:.0f}'inde drawdown sermayenin "
            f"%{mc.dd_limit_pct:.0f}'ini aşıyor.")
    return Verdict(passed=edge_proven, headline=head, notes=notes)


def _fmt_i(iv: Interval, spec: str = "+.3f") -> str:
    return (f"{iv.point:{spec}}  [95% GA: {iv.lo:{spec}} … {iv.hi:{spec}}]")


def format_validation(trades: list[Trade], boot: BootstrapResult,
                      mc: MonteCarloResult, vd: Verdict) -> str:
    setups = sorted({t.setup for t in trades})
    days = sorted({t.day for t in trades if t.day})
    lines = [
        "=== İSTATİSTİKSEL DOĞRULAMA ===",
        f"  Örneklem : {boot.n_trades} işlem | {len(days)} seans"
        f"{f' ({days[0]} → {days[-1]})' if days else ''}",
        f"  Setup    : {', '.join(setups)}",
        f"  Yöntem   : {boot.iterations:,} bootstrap + {mc.iterations:,} monte carlo"
        f" (seed sabit, tekrarlanabilir)",
        "",
        "  --- Bootstrap (edge gerçek mi?) ---",
        f"  Beklenti (R/işlem) : {_fmt_i(boot.expectancy_r)}",
        f"  Profit factor      : {_fmt_i(boot.profit_factor, '.2f')}",
        f"  Win rate           : {_fmt_i(boot.win_rate, '.1%')}",
        f"  Edge yok olasılığı : %{boot.p_no_edge*100:.1f}"
        "   (bootstrap dağılımının sıfır altında kalan oranı)",
    ]
    if boot.trades_needed:
        lines.append(f"  Gereken örneklem   : ~{boot.trades_needed:,} işlem "
                     f"(bu etki büyüklüğü için)")
    lines += [
        "",
        "  --- Monte Carlo (aynı strateji, alternatif koşular) ---",
        f"  Gerçekleşen PnL    : ${mc.realized_pnl:+,.2f}",
        f"  Koşu sonu PnL      : {_fmt_i(mc.final_pnl, ',.2f')}",
        f"  Gerçekleşen max DD : ${mc.realized_max_dd:,.2f}",
        f"  Max DD dağılımı    : {_fmt_i(mc.max_dd, ',.2f')}",
        f"  Sadece sıra değişse: {_fmt_i(mc.shuffle_max_dd, ',.2f')}",
        f"  Zararla bitme      : %{mc.p_losing_run*100:.1f}",
        f"  DD > sermaye %{mc.dd_limit_pct:.0f}   : %{mc.p_deep_dd*100:.1f}",
        "",
        f"  >>> {vd.headline}",
    ]
    lines += [f"      • {n}" for n in vd.notes]
    return "\n".join(lines)


def validate(trades: list[Trade], start_equity: float,
             iterations: int = DEFAULT_ITERATIONS, seed: int = DEFAULT_SEED,
             pf_target: float = 1.3) -> tuple[BootstrapResult, MonteCarloResult,
                                              Verdict, str]:
    """Run the full validation suite and return results plus a printable report."""
    boot = bootstrap(trades, iterations=iterations, seed=seed)
    mc = monte_carlo(trades, start_equity, iterations=iterations, seed=seed)
    vd = verdict(boot, mc, pf_target=pf_target)
    return boot, mc, vd, format_validation(trades, boot, mc, vd)
