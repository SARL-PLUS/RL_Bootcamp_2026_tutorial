#!/usr/bin/env python
"""Render :mod:`envs.flight_4d` episodes to video, one agent, or several.

    conda activate rlbootcamp
    python scripts/render_4d.py --agent noop --agent rule-based --seed 900000

    # the action vocabulary, one clearance per clip, two aircraft and nothing else
    python scripts/render_4d.py --n-flights 2 --n-range 2 2 --seed 2 --ext gif \
        --agent noop --agent fl_inc --agent fl_inc2_t --agent spd_dn_t --agent resume

``Flight4DEnv`` itself declares no render modes (``metadata = {"render_modes":
[]}``) — hard masking and a flat action space were the design problem, not
visualisation, and the environment stays free of matplotlib. This script
drives the environment from the outside: step it, record what happened, then
replay the recording through :class:`envs.simulator.CollisionCourseSimulator`,
which both this and the retired environment share and which already carries
every geometry primitive rendering needs (``proximity_pairs``,
``find_congestion``, ``get_alt_color``).

The congestion styling — dashed while a pair is still legally separated, solid
once it is not, linewidth and opacity rising with severity, and a fading trail
so a resolved conflict does not vanish the instant the aircraft move on — is
carried over from the retired environment's own renderer, where it replaced an
almost-invisible ``alpha=0.1`` fixed circle. The visibility problem was solved
once; there was no reason to solve it twice.

Two things are drawn per aircraft that were not in that original renderer.
**The flight plan itself**: a solid track behind each aircraft, one colour per
segment for the level actually held while flying it, and a dashed projection
ahead, one colour for the level held right now, running exactly as far as the
aircraft still has left to fly (``s_left``) — since headings are never
commanded, that projection is not a guess, it is where the aircraft goes if
nothing else touches it. **The two KPI bars**, safety and timeliness, read
straight off the same ``_severity``/``_deviation`` methods the reward uses, on
a green-red scale where low is good — a render that agreed with the reward
only approximately would be a second source of error, not a picture of it.
"""
from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import Noop4DController, Priority4DController          # noqa: E402
from envs.flight_4d import CLEARANCE_NAMES, N_CLEARANCES, Flight4DEnv  # noqa: E402

#: Low is good, high is bad, for both KPI bars and the per-aircraft deviation
#: tick — safety (severity) and timeliness (4D deviation) are both costs.
GOOD_BAD = LinearSegmentedColormap.from_list("good_bad", ["#2ca02c", "#d62728"])

#: One marker + colour per clearance, grouped by what it does to the aircraft.
#: Altitude commands are triangles (direction = up/down), speed commands are
#: sideways triangles, RESUME is the only circle — "coming home" reads as a
#: full stop, not a further manoeuvre.
CLEARANCE_STYLE = {
    "FL_DEC":    dict(marker="v", color="#d62728"),   # descend
    "FL_INC":    dict(marker="^", color="#1f77b4"),   # climb
    "FL_DEC2_T": dict(marker="v", color="#7f0d0d"),   # descend, two levels
    "FL_INC2_T": dict(marker="^", color="#0d3d7f"),   # climb, two levels
    "SPD_DN_T":  dict(marker="<", color="#17becf"),   # slow down
    "SPD_UP_T":  dict(marker=">", color="#ff7f0e"),   # speed up
    "RESUME":    dict(marker="o", color="#2ca02c"),   # resume own navigation
}


#: Scripted demonstrations, for the handbook's "what a clearance buys you"
#: figures. Each is a fixed list of ``(step, aircraft, clearance)`` on ``DEMO_SEED``
#: at ``n_range=(2, 2)``, so the aircraft are a single converging pair with
#: nothing else at their level.
#:
#: The steps are chosen against that seed's geometry rather than derived: flights
#: 0 and 1 are co-level at FL374 and close to d_cpa = 1.0 m at **step 18**. The
#: clearances are therefore an approximation, which is the honest thing to
#: demonstrate --- the action set is discrete, one aircraft may be commanded per
#: step, and a temporary manoeuvre's return leg is on a timer the caller does not
#: control. Timing a resolution is part of the problem, not an implementation
#: detail to hide.
DEMO_SEED = 2
DEMO_CPA_STEP = 18
#: The step every single-clearance demo issues its one instruction on: four
#: steps (8 s) before closest approach, late enough that the macro's hold
#: window straddles the CPA and early enough for a speed pulse to open a gap.
DEMO_ACT_STEP = 14

DEMOS = {
    # Two permanent, opposite level changes open ALT_MIN_SEP between the pair and
    # hold it open. RESUME comes AFTER the closest approach, chosen by the
    # controller rather than by a timer -- the pattern the rule-based baseline uses.
    "split": [(6, 0, "FL_INC"), (7, 1, "FL_DEC"),
              (24, 0, "RESUME"), (25, 1, "RESUME")],
    # One instruction buys two levels AND books its own return. Issued late so the
    # hold window straddles the CPA -- exactly the transient encounter the macro is
    # the right tool for. Watch `hold_left` count down and the aircraft come home
    # on its own, with no second clearance.
    "macro": [(14, 0, "FL_INC2_T")],
    # A symmetric speed pulse on each aircraft, opposite signs: one arrives at the
    # hotspot late, the other early, so they are never there together. Both halves
    # of each pulse are inside one clearance, so the schedule is returned intact --
    # which is what makes this the 4D-neutral way to resolve a conflict.
    "speed": [(4, 0, "SPD_DN_T"), (5, 1, "SPD_UP_T")],
    # --- one clearance each, same aircraft, same step: the action vocabulary
    # shown one instruction at a time. Meant for ``--n-flights 2 --n-range 2 2``
    # so nothing else is on screen. Measured outcomes on DEMO_SEED (noop
    # collides at step 18 with 3 congestion events):
    #   FL_INC / FL_DEC       no mid-air, 5 congestion events, exit_miss 0.5
    #                         -- one level is NOT a fix, and it never comes home
    #   FL_INC2_T / FL_DEC2_T 2 congestion events, exit_miss 0 -- two levels
    #                         resolve it, the revert is booked with the order
    #   SPD_DN_T / SPD_UP_T   258-274 m miss, 4-5 congestion events, exit_miss 0
    #                         -- 70 m per step against a 1000 m radius
    #   RESUME                FL_INC then RESUME ten steps later: home, exit_miss 0
    "fl_inc":    [(DEMO_ACT_STEP, 0, "FL_INC")],
    "fl_dec":    [(DEMO_ACT_STEP, 0, "FL_DEC")],
    "fl_inc2_t": [(DEMO_ACT_STEP, 0, "FL_INC2_T")],
    "fl_dec2_t": [(DEMO_ACT_STEP, 0, "FL_DEC2_T")],
    "spd_dn_t":  [(DEMO_ACT_STEP, 0, "SPD_DN_T")],
    "spd_up_t":  [(DEMO_ACT_STEP, 0, "SPD_UP_T")],
    "resume":    [(DEMO_ACT_STEP, 0, "FL_INC"), (DEMO_ACT_STEP + 10, 0, "RESUME")],
}


class ScriptedController:
    """Replay a fixed ``(step, aircraft, clearance)`` list; NOOP otherwise.

    Deliberately open-loop. These clips exist to show what ONE clearance does to
    the geometry, and a reactive controller would make it impossible to say which
    part of the outcome came from which decision.
    """

    def __init__(self, script):
        self.script = {s: (i, c) for s, i, c in script}
        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def predict(self, obs, deterministic: bool = True):
        entry = self.script.get(self.step)
        self.step += 1
        if entry is None:
            return 0, None
        i, name = entry
        return 1 + i * N_CLEARANCES + CLEARANCE_NAMES.index(name), None


def load_agent(spec: str, env: Flight4DEnv):
    if spec in DEMOS:
        return ScriptedController(DEMOS[spec])
    if spec == "noop":
        return Noop4DController()
    if spec == "random":
        return None  # handled inline: uniform over the legal actions
    if spec == "rule-based":
        return Priority4DController(n_flights=env.n_flights, max_steps=env.max_steps)
    from stable_baselines3 import PPO
    return PPO.load(spec, device="cpu")


def rollout(spec: str, seed: int, env_kwargs: dict, deterministic: bool = True):
    """Step the environment, recording exactly what the render needs.

    Severity and per-aircraft deviation are read straight off the same methods
    the reward uses (``_severity``, ``_deviation``), not re-derived from the
    recorded positions afterwards — a render that agreed with the reward only
    approximately would be a second source of error, not a picture of it.

    Returns the environment (post-episode, so its final ``sim`` seeds the
    colour scale and axis limits) plus one entry per step:
    ``(snapshot, clearance, congestion_now, severity_now, deviation_now,
    s_left)`` where ``snapshot`` is ``[(x, y, alt, vel, hdg), ...]``,
    ``clearance`` is ``None`` or ``(aircraft_index, clearance_name)``,
    ``deviation_now`` and ``s_left`` are per-aircraft arrays (the latter is
    the remaining distance to each aircraft's own 4D exit point, for the
    predicted-track line).
    """
    env = Flight4DEnv(**env_kwargs)
    obs, _ = env.reset(seed=seed)
    agent = load_agent(spec, env)
    if hasattr(agent, "reset"):
        agent.reset()
    rng = np.random.default_rng(seed)

    def snapshot():
        return [(f.x, f.y, f.altitude, f.velocity, f.heading) for f in env.sim.flights]

    def s_left():
        return np.maximum(env.s_total - env.s_flown, 0.0)

    frames = [(snapshot(), None, 0, env._severity(), env._deviation(), s_left())]
    while True:
        if spec == "random":
            action = int(rng.choice(np.flatnonzero(env.action_masks())))
        else:
            action = int(agent.predict(obs, deterministic=deterministic)[0])
        clearance = None
        if action > 0:
            i, cmd = divmod(action - 1, N_CLEARANCES)
            clearance = (i, CLEARANCE_NAMES[cmd])
        obs, _, term, trunc, info = env.step(action)
        frames.append((snapshot(), clearance, info["congestion_events"],
                      env._severity(), env._deviation(), s_left()))
        if term or trunc:
            break
    return env, frames


def render(env: Flight4DEnv, frames: list, title: str, save_path: Path,
           fps: int = 5, trail_halflife: float = 4.0, trail_floor: float = 0.04,
           clearance_halflife: float = 6.0,
           safety_cap: float = 2.0) -> None:
    sim = deepcopy(env.sim)
    hl = sim.hotspot_limit

    # Main plot | safety bar | timeliness bar | altitude colorbar. width_ratios
    # leave the main axes enough room to render as a true square once
    # aspect='equal' pins its data limits — the two KPI bars and the colorbar
    # only need a sliver each.
    fig = plt.figure(figsize=(9.4, 7.4))
    gs = fig.add_gridspec(1, 4, width_ratios=(10, 0.7, 0.7, 0.5), wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    ax_safe = fig.add_subplot(gs[0, 1])
    ax_time = fig.add_subplot(gs[0, 2])
    ax_cbar = fig.add_subplot(gs[0, 3])

    alt_norm = mpl.colors.Normalize(
        vmin=sim.ALT_MIN, vmax=sim.ALT_MIN + sim.DALT * len(sim.allowable_altitudes))
    sm = mpl.cm.ScalarMappable(cmap=mpl.cm.plasma, norm=alt_norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cbar, ticks=sim.allowable_altitudes, label="Altitude")

    lim = sim.airspace_limit * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")   # the plot area itself is square
    ax.add_patch(Rectangle((-hl, -hl), 2 * hl, 2 * hl, alpha=0.7, fill=False, color="k"))

    # Legend outside the plot (below it) so it never sits over the traffic.
    for name, style in CLEARANCE_STYLE.items():
        ax.scatter([], [], s=24, marker=style["marker"], color=style["color"], label=name)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False,
             fontsize="x-small", ncol=4)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.14)

    # --- the two KPI bars: low is good (green), high is bad (red) -----------
    # Labels go UNDER each bar, not up its y-axis: a rotated ylabel on a
    # sliver-wide axis lands over the main plot on one side and over the
    # neighbouring bar on the other. The tick labels are pushed outwards for
    # the same reason -- the safety bar reads on its left, the timeliness bar
    # on its right -- so nothing sits in the gap between the two bars.
    safe_bar = ax_safe.bar([0], [0.0], width=0.6, color=GOOD_BAD(0.0))[0]
    time_bar = ax_time.bar([0], [0.0], width=0.6, color=GOOD_BAD(0.0))[0]
    for a, cap, xlabel in ((ax_safe, safety_cap, "safety\n(severity)"),
                           (ax_time, 1.0, "timeliness\n(mean dev.)")):
        a.set_xlim(-0.5, 0.5)
        a.set_ylim(0, cap)
        a.set_xticks([])
        a.set_xlabel(xlabel, fontsize=7, labelpad=4)
        a.tick_params(labelsize=7)
    ax_time.yaxis.tick_right()

    arrow_size = sim.airspace_limit * 0.035
    text_dy = sim.airspace_limit * 0.035    # FL/V label, above the aircraft
    dev_dy = sim.airspace_limit * 0.16      # deviation tick, clear of the label
    dev_halfwidth = sim.airspace_limit * 0.045
    arrows, fl_texts, dev_ticks = [], [], []
    #: One fading LineCollection per aircraft for the flown track, plus one
    #: dashed Line2D for the projected remainder — recreated every frame.
    flown_tracks: list = [None] * env.n_flights
    predicted_tracks: list = [None] * env.n_flights
    for flight in sim.flights:
        c = sim.get_alt_color(flight.altitude)
        arrows.append(mpl.patches.FancyArrow(
            flight.x, flight.y, 0, 0, edgecolor=c, facecolor=c,
            head_width=arrow_size, head_length=arrow_size, zorder=15))
        fl_texts.append(ax.text(0, 0, "", fontsize=7))
        dev_ticks.append(ax.plot([], [], lw=2.5, solid_capstyle="butt", zorder=16)[0])

    live_circles: list = []
    trail: list = []
    clearance_markers: list = []
    max_cong = 0

    def apply_frame(k: int) -> None:
        snap, *_ = frames[min(k, len(frames) - 1)]
        for f, (x, y, alt, vel, hdg) in zip(sim.flights, snap):
            f.x, f.y, f.altitude, f.velocity, f.heading = x, y, alt, vel, hdg

    def promote_and_age(frame: int) -> None:
        """Retire this frame's circles; keep only past REAL conflicts, fading."""
        while live_circles:
            circle, in_conflict = live_circles.pop()
            if in_conflict:
                trail.append((circle, frame, circle.get_alpha(), circle.get_linewidth()))
            else:
                circle.remove()
        survivors = []
        for circle, born, base_alpha, base_lw in trail:
            fade = 0.5 ** ((frame - born) / max(trail_halflife, 1e-6))
            if base_alpha * fade < trail_floor:
                circle.remove()
                continue
            circle.set_alpha(base_alpha * fade)
            circle.set_linewidth(base_lw * (0.35 + 0.65 * fade))
            circle.set_zorder(10)
            survivors.append((circle, born, base_alpha, base_lw))
        trail[:] = survivors

    def init():
        while live_circles:
            live_circles.pop()[0].remove()
        while trail:
            trail.pop()[0].remove()
        while clearance_markers:
            clearance_markers.pop()[0].remove()
        for i in range(env.n_flights):
            if flown_tracks[i] is not None:
                flown_tracks[i].remove()
                flown_tracks[i] = None
            if predicted_tracks[i] is not None:
                predicted_tracks[i].remove()
                predicted_tracks[i] = None
        apply_frame(0)
        for i, flight in enumerate(sim.flights):
            dx, dy = flight.get_heading_unit_vectors()
            arrows[i] = mpl.patches.FancyArrow(
                flight.x, flight.y, dx, dy, width=0.01, color="b")
            ax.add_patch(arrows[i])
            fl_texts[i].set_text("")
            dev_ticks[i].set_data([], [])
        ax.set_title(title + "\nStep=0")
        return fl_texts + arrows

    def age_clearance_markers(frame: int) -> None:
        """Fade past clearance markers instead of either wiping them
        instantly (loses the pattern of where interventions concentrated) or
        keeping them forever (a 30+-clearance episode turns one hotspot into
        an unreadable pile, since a marker stamps the aircraft's position at
        the moment it was commanded and does not move with it afterwards)."""
        survivors = []
        for marker, born, base_alpha in clearance_markers:
            fade = 0.5 ** ((frame - born) / max(clearance_halflife, 1e-6))
            if base_alpha * fade < trail_floor:
                marker.remove()
                continue
            marker.set_alpha(base_alpha * fade)
            survivors.append((marker, born, base_alpha))
        clearance_markers[:] = survivors

    def update(frame: int):
        nonlocal max_cong
        apply_frame(frame)
        _, clearance, cong_now, severity_now, deviation_now, s_left_now = frames[frame]
        max_cong = max(max_cong, cong_now)
        promote_and_age(frame)
        age_clearance_markers(frame)

        # --- the two KPI bars: this frame's cost, not the episode running sum
        safe_v = min(severity_now, safety_cap)
        safe_bar.set_height(safe_v)
        safe_bar.set_color(GOOD_BAD(safe_v / safety_cap))
        time_v = float(np.clip(deviation_now.mean(), 0.0, 1.0))
        time_bar.set_height(time_v)
        time_bar.set_color(GOOD_BAD(time_v))

        for cen, calt, resolved, severity, in_conflict in sim.proximity_pairs(sim.flights):
            circle = plt.Circle(
                cen, max(resolved, 0.05) * hl, color=sim.get_alt_color(calt),
                fill=False, alpha=0.30 + 0.55 * severity, lw=0.4 + 3.0 * severity,
                ls="-" if in_conflict else (0, (4, 3)), zorder=20)
            ax.add_patch(circle)
            live_circles.append((circle, in_conflict))

        for i, flight in enumerate(sim.flights):
            # Flown track: solid, thin, one colour per segment — the level
            # actually held while flying that stretch, not the current one.
            if flown_tracks[i] is not None:
                flown_tracks[i].remove()
            pts = [(frames[k][0][i][0], frames[k][0][i][1]) for k in range(frame + 1)]
            if len(pts) >= 2:
                segs = [[pts[k], pts[k + 1]] for k in range(len(pts) - 1)]
                cols = [sim.get_alt_color(frames[k][0][i][2]) for k in range(len(pts) - 1)]
                flown_tracks[i] = LineCollection(segs, colors=cols, linewidths=1.1, zorder=12)
                ax.add_collection(flown_tracks[i])
            else:
                flown_tracks[i] = None

            if not sim.flight_within_limit(flight):
                continue
            ux, uy = flight.get_heading_unit_vectors()
            c = sim.get_alt_color(flight.altitude)

            # Predicted track: dashed, thin, single colour (this frame's level)
            # — the straight-line projection to this aircraft's own 4D exit
            # point, exactly as far as it still has left to fly.
            if predicted_tracks[i] is not None:
                predicted_tracks[i].remove()
            reach = float(s_left_now[i])
            (predicted_tracks[i],) = ax.plot(
                [flight.x, flight.x + ux * reach], [flight.y, flight.y + uy * reach],
                lw=1.0, ls=(0, (5, 4)), color=c, zorder=12)

            arrows[i].remove()
            arrows[i] = mpl.patches.FancyArrow(
                flight.x, flight.y, ux * arrow_size, uy * arrow_size,
                edgecolor=c, facecolor=c, head_width=arrow_size,
                head_length=arrow_size, zorder=15)
            ax.add_patch(arrows[i])
            fl_texts[i].remove()
            fl_texts[i] = ax.text(flight.x, flight.y + text_dy,
                                  f"FL{flight.altitude:.0f}\nV{flight.velocity:.0f}",
                                  fontsize=7, ha="center", va="bottom")

            # 4D-exit deviation, one short tick per aircraft, same green-red
            # scale as the KPI bars so a glance at either reads the same way.
            dev = float(np.clip(deviation_now[i], 0.0, 1.0))
            yy = flight.y + dev_dy
            dev_ticks[i].set_data([flight.x - dev_halfwidth, flight.x + dev_halfwidth],
                                  [yy, yy])
            dev_ticks[i].set_color(GOOD_BAD(dev))

            if clearance is not None and clearance[0] == i:
                style = CLEARANCE_STYLE[clearance[1]]
                marker = ax.scatter([flight.x], [flight.y], s=60, marker=style["marker"],
                                    color=style["color"], zorder=25, alpha=1.0,
                                    edgecolors="k", linewidths=0.5)
                clearance_markers.append((marker, frame, 1.0))

        ax.set_title(f"{title}\nStep={frame}  Conflicts now={cong_now}  "
                     f"Max={max_cong}  Clearances={env.n_clearances}  "
                     f"Invalids={env.n_invalids}")
        return fl_texts + arrows + dev_ticks

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  init_func=init, blit=False, repeat=False)
    # Writer follows the extension. GIF is what the handbook embeds -- it plays
    # in every browser with no plugin, where an .mp4 would need one and is
    # gitignored besides. It costs bytes, so it renders at a lower dpi.
    if save_path.suffix.lower() == ".gif":
        ani.save(str(save_path), dpi=72,
                 writer=animation.PillowWriter(fps=fps))
    else:
        ani.save(str(save_path), dpi=110,
                 writer=mpl.animation.FFMpegWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", action="append", required=True,
                   help="'noop', 'random', 'rule-based', a scripted demo "
                        f"({', '.join(DEMOS)}), or a path to a PPO .zip; repeatable")
    p.add_argument("--seed", type=int, default=900_000,
                   help="900000 is the first of score_4d.py's held-out block, "
                        "and draws n=3 (m=2) at the environment's defaults")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of taking the argmax")
    p.add_argument("--out", type=Path, default=Path("renders"))
    p.add_argument("--fps", type=int, default=5)
    p.add_argument("--ext", choices=("mp4", "gif"), default="mp4",
                   help="gif for the handbook (plays anywhere, no plugin); "
                        "mp4 for the decks")
    p.add_argument("--n-range", type=int, nargs=2, default=None,
                   metavar=("LO", "HI"),
                   help="pin the converging-traffic draw, e.g. --n-range 2 2 "
                        "for the two-aircraft demonstrations")
    p.add_argument("--n-flights", type=int, default=None,
                   help="sector size n + m (default: the environment's 5). "
                        "--n-flights 2 --n-range 2 2 puts exactly two aircraft "
                        "on screen and nothing else -- the per-action demos")
    args = p.parse_args()

    env_kwargs = {}
    if args.n_range is not None:
        env_kwargs["n_range"] = tuple(args.n_range)
    if args.n_flights is not None:
        env_kwargs["n_flights"] = args.n_flights
    args.out.mkdir(parents=True, exist_ok=True)
    for spec in args.agent:
        known = ("noop", "random", "rule-based", *DEMOS)
        label = spec if spec in known else Path(spec).stem
        env, frames = rollout(spec, args.seed, env_kwargs,
                              deterministic=not args.stochastic)
        n, m = env.n, env.m
        title = f"{label} — n={n} m={m}, seed={args.seed}"
        out_path = args.out / f"seed{args.seed}_n{n}m{m}_{label}.{args.ext}"
        render(env, frames, title, out_path, fps=args.fps)
        print(f"  {label:12s} n={n} m={m}  steps={env.step_idx:2d}  "
              f"collided={env.collided}  congestion_total={env.congestion_total:3d}  "
              f"clearances={env.n_clearances:3d}  exit_miss={env._info()['exit_miss']:.3f}"
              f"  -> {out_path}")


if __name__ == "__main__":
    main()
