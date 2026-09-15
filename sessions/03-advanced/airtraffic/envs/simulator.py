import math
import random
import string
from collections import namedtuple
from copy import deepcopy

import numpy as np
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from .flight import Flight


def _random_alphanum(n=4):
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))


class CollisionCourseSimulator:
    """
    Physics engine for the AirTraffic environment.

    Initialises n flights on a collision course converging on a hotspot,
    plus m random background flights. Provides clock stepping, collision
    (congestion) detection, normalisation helpers, save/load, and animation.
    """

    hotspot2airspace_factor  = 0.1
    spawn2airspace_factor    = 0.9
    stagger2airspace_factor  = 0.001

    DT       = 2      # time step
    DALT     = 12     # FL step
    ALT_MIN  = 290    # minimum FL
    N_ALTS   = 10
    ALT_MIN_SEP = DALT * 2   # horizontal separation threshold (altitude)
    # Raised from 10 on 2026-08-20. At 10 m/s a velocity clearance opened 20 m of
    # along-track offset per step against a 1000 m conflict radius — 50 steps of a
    # 50-step episode — so speed was structurally incapable of resolving a conflict
    # and neither the trained agent nor the hand-written controller ever issued one.
    # 35 makes it 70 m/step. Note this widens the speed range to 200-515 m/s.
    DVEL     = 35     # velocity step (m/s)
    VEL_MIN  = 200    # minimum velocity (m/s)
    N_VELS   = 10
    AIRSPACE_LIMIT = 10000

    def __init__(self, n, m, airspace_limit=AIRSPACE_LIMIT, init_flights=True, rng=None):
        self.n = n
        self.m = m
        # All stochastic initialisation draws from this generator, so a
        # seeded rng makes the whole scenario reproducible.
        self.rng = rng if rng is not None else np.random.default_rng()
        self.airspace_limit = airspace_limit
        self.hotspot_limit  = airspace_limit * self.hotspot2airspace_factor
        self.collision_x    = self.rng.uniform(-self.hotspot_limit, self.hotspot_limit)
        self.collision_y    = self.rng.uniform(-self.hotspot_limit, self.hotspot_limit)

        self.allowable_altitudes  = self.get_allowable_altitudes()
        self.allowable_velocities = self.get_allowable_velocities()

        self.flights     = []
        self.sim_time    = 0
        self.evo_history = {}

        if init_flights:
            self.init_flights()

    # ------------------------------------------------------------------
    # Normalisation / denormalisation
    # ------------------------------------------------------------------

    def get_allowable_altitudes(self):
        return np.arange(self.N_ALTS) * self.DALT + self.ALT_MIN

    def get_allowable_velocities(self):
        return np.arange(self.N_VELS) * self.DVEL + self.VEL_MIN

    def norm_pos(self, pos):
        return pos / self.airspace_limit

    def norm_vel(self, vel):
        return (vel - self.VEL_MIN) / ((self.N_VELS - 1) * self.DVEL)

    def norm_alt(self, alt):
        return (alt - self.ALT_MIN) / ((self.N_ALTS - 1) * self.DALT)

    def norm_heading(self, heading):
        return heading / 360.

    def denorm_pos(self, pos):
        return pos * self.airspace_limit

    def denorm_vel(self, vel):
        return vel * (self.N_VELS - 1) * self.DVEL + self.VEL_MIN

    def denorm_alt(self, alt):
        return alt * (self.N_ALTS - 1) * self.DALT + self.ALT_MIN

    def denorm_heading(self, heading):
        return heading * 360.

    # ------------------------------------------------------------------
    # Flight data accessors
    # ------------------------------------------------------------------

    def get_flight_data(self, idx=None, flight=None, norm=True):
        assert (idx is None) != (flight is None), 'Provide exactly one of idx or flight.'
        if idx is not None:
            assert idx < len(self.flights), f'Index {idx} out of range.'
            flight = self.flights[idx]

        if isinstance(flight, Flight):
            x, y, alt, vel, heading = flight.get_5_tuple()
        elif isinstance(flight, (list, np.ndarray, tuple)):
            x, y, alt, vel, heading = flight
        elif isinstance(flight, dict):
            x, y, alt, vel, heading = [flight[k] for k in Flight.TUPLE5KEYS]
        else:
            raise NotImplementedError(f'Unsupported flight type: {type(flight)}')

        if norm:
            x       = self.norm_pos(x)
            y       = self.norm_pos(y)
            alt     = self.norm_alt(alt)
            vel     = self.norm_vel(vel)
            heading = self.norm_heading(heading)

        return x, y, alt, vel, heading

    def get_denorm_flight_data(self, idx=None, flight=None):
        return self.get_flight_data(idx=idx, flight=flight, norm=False)

    def get_flight_data_prev_timestep(self, idx, norm=True):
        if len(self.evo_history) <= 1:
            return self.get_flight_data(idx=idx, norm=norm)
        prev_time = self.sim_time - self.DT
        if prev_time not in self.evo_history:
            raise KeyError(f'No history at prev_time={prev_time}.')
        tuple_5 = self.flights[idx].evo_history[prev_time]
        return self.get_flight_data(flight=tuple_5, norm=norm)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_flights(self):
        init_altitudes = self.allowable_altitudes.copy()
        coll_idx = self.rng.choice(len(init_altitudes))
        collision_altitude = init_altitudes[coll_idx]
        init_altitudes = np.delete(init_altitudes, coll_idx)

        # n flights converging on hotspot (same starting altitude)
        for _ in range(self.n):
            heading  = self.rng.uniform(0, 360)
            velocity = self.rng.choice(self.allowable_velocities)
            flight   = Flight(altitude=collision_altitude, velocity=velocity, heading=heading,
                              x=self.collision_x, y=self.collision_y,
                              VEL_MIN=self.VEL_MIN, ALT_MIN=self.ALT_MIN)
            self.flights.append(flight)
        self.flights = self.reverse_to_limit(self.flights)
        self.flights = self.stagger_flights(self.flights)

        # m random background flights
        spawn_limit = self.airspace_limit * self.spawn2airspace_factor
        for _ in range(self.m):
            x, y = self.collision_x, self.collision_y
            while self._is_near_collision(x, y):
                x = self.rng.uniform(-spawn_limit, spawn_limit)
                y = self.rng.uniform(-spawn_limit, spawn_limit)
            passive_altitude = self.rng.choice(init_altitudes)
            velocity         = self.rng.choice(self.allowable_velocities)
            heading          = self.rng.uniform(0, 360)
            flight = Flight(altitude=passive_altitude, velocity=velocity, heading=heading,
                            x=x, y=y, VEL_MIN=self.VEL_MIN, ALT_MIN=self.ALT_MIN)
            flight = self.reverse_to_limit([flight])[0]
            for _ in range(self.rng.choice(10)):
                flight.reverse_clock(self.DT)
            self.flights.append(flight)

        self.advance_into_airspace()

    def advance_into_airspace(self, max_shift=200):
        """Uniformly time-shift the scenario until every flight is inside the box.

        ``reverse_to_limit`` deliberately rewinds every flight until it is *outside*
        the airspace, which leaves them invisible at t=0: positions are normalised by
        ``airspace_limit``, so an aircraft that has not arrived yet reads outside
        [-1, 1] and the agent is asked to plan around traffic it cannot properly see.

        The shift is **uniform** — every flight advances together — so relative
        geometry and arrival order are untouched. Only the origin of the clock moves,
        to the first moment the whole picture is on screen.
        """
        for _ in range(max_shift):
            if all(self.flight_within_limit(f) for f in self.flights):
                return
            for f in self.flights:
                f.advance_clock(self.DT)
            self.sim_time += self.DT

    def _is_near_collision(self, x, y):
        return math.hypot(x - self.collision_x, y - self.collision_y) < self.hotspot_limit

    # ------------------------------------------------------------------
    # Clock stepping
    # ------------------------------------------------------------------

    def flight_within_limit(self, flight):
        return max(abs(flight.x), abs(flight.y)) < self.airspace_limit

    def time_to_exit(self, flight):
        """Seconds until `flight` leaves the airspace on its current track.

        The airspace is the square max(|x|, |y|) < airspace_limit and headings are
        never commanded, so this is a ray-vs-box problem solvable in closed form.
        Uses the slab method rather than "first wall crossing ahead", because
        flights **spawn outside the sector and fly in** — for those, the first
        crossing ahead is the *entry*, not the exit, and reporting it would make
        an aircraft that has not arrived yet look like one about to leave.

        Returns the exit time for a flight inside the sector or approaching it,
        and 0.0 for one that has already left or is heading away.
        """
        ux, uy = flight.get_heading_unit_vectors()
        speed = flight.velocity
        if speed <= 0:
            return 0.0

        lim = self.airspace_limit
        t_enter, t_exit = -math.inf, math.inf

        for pos, u in ((flight.x, ux), (flight.y, uy)):
            if abs(u) < 1e-12:
                # Parallel to this pair of walls: either always between them or
                # never, and no crossing time exists either way.
                if abs(pos) >= lim:
                    return 0.0
                continue
            t1 = (-lim - pos) / (u * speed)
            t2 = (lim - pos) / (u * speed)
            lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
            t_enter = max(t_enter, lo)
            t_exit = min(t_exit, hi)

        if t_enter > t_exit or t_exit <= 0:
            return 0.0          # never crosses the sector, or already past it
        return float(t_exit)

    def reverse_to_limit(self, flights):
        """Reverse all flights until every one has left the airspace, then step forward 2."""
        done = False
        while not done:
            done = all(not self.flight_within_limit(f) for f in flights)
            for f in flights:
                f.reverse_clock(self.DT)
        for f in flights:
            for _ in range(2):
                f.advance_clock(self.DT)
        return flights

    def advance_clock(self):
        for flight in self.flights:
            flight.advance_clock(self.DT)
        self.sim_time += self.DT
        self.evo_history[self.sim_time] = [f.time for f in self.flights]

    def reverse_clock(self):
        for flight in self.flights:
            flight.reverse_clock(self.DT)
        self.sim_time -= self.DT
        self.evo_history[self.sim_time] = [f.time for f in self.flights]

    def stagger_flights(self, flights):
        stagger = self.airspace_limit * self.stagger2airspace_factor
        for flight in flights:
            flight.x += self.rng.uniform(-stagger, stagger)
            flight.y += self.rng.uniform(-stagger, stagger)
        return flights

    # ------------------------------------------------------------------
    # Collision / congestion detection
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def is_within_altitude_range(self, alt1, alt2):
        return abs(alt1 - alt2) < self.ALT_MIN_SEP

    def find_congestion(self, flights):
        """Return (centers, altitudes) of congestion groups (potential collisions)."""
        n = len(flights)
        xs   = [f.x        for f in flights]
        ys   = [f.y        for f in flights]
        alts = [f.altitude for f in flights]

        congestion_centers   = []
        congestion_altitudes = []

        for i in range(n):
            nearby_xy  = [[xs[i], ys[i]]]
            nearby_alt = [alts[i]]
            for j in range(i + 1, n):
                if self.is_within_altitude_range(alts[i], alts[j]):
                    if self.calculate_distance(xs[i], ys[i], xs[j], ys[j]) <= self.hotspot_limit:
                        nearby_alt.append(alts[j])
                        nearby_xy.append([xs[j], ys[j]])
            if len(nearby_alt) >= 2:
                congestion_centers.append(np.mean(np.array(nearby_xy), axis=0))
                congestion_altitudes.append(np.mean(nearby_alt))

        return congestion_centers, congestion_altitudes

    #: How far out, in resolved units, the renderer draws proximity circles.
    #: 1.0 is exactly the conflict threshold, so 2.5 shows pairs closing on it as
    #: well as pairs already inside it.
    PROXIMITY_REACH = 2.5

    def proximity_pairs(self, flights=None, reach=None):
        """Pairwise closeness right now, for rendering. Not a KPI.

        ``find_congestion`` answers a yes/no question — is this pair inside the
        box ``horizontal <= hotspot_limit`` AND ``|dalt| < ALT_MIN_SEP``. Drawing
        only that gives every conflict the same fixed circle, so a pair scraping
        the threshold looks identical to two aircraft about to occupy the same
        point, and a pair one metre outside it vanishes entirely.

        The **resolved distance** normalises each axis by its own threshold and
        takes the larger:

            resolved = max(horizontal / hotspot_limit, |dalt| / ALT_MIN_SEP)

        The max, not the hypotenuse, because the conflict condition is a *box*:
        ``resolved < 1`` is then exactly ``find_congestion``'s test, with no
        approximation. Below 1 the pair is in conflict and the number says how
        deeply; above 1 it says how much margin is left.

        Returns one tuple per pair within ``reach``::

            (centre_xy, mean_altitude, resolved, severity, in_conflict)

        ``severity`` is ``1 - resolved / reach`` clipped to [0, 1], so it rises
        smoothly as a pair closes and is largest for the pairs that matter most.
        """
        flights = self.flights if flights is None else flights
        reach = self.PROXIMITY_REACH if reach is None else float(reach)
        out = []
        for i in range(len(flights)):
            fi = flights[i]
            for j in range(i + 1, len(flights)):
                fj = flights[j]
                d_h = self.calculate_distance(fi.x, fi.y, fj.x, fj.y) / self.hotspot_limit
                d_v = abs(fi.altitude - fj.altitude) / self.ALT_MIN_SEP
                resolved = max(d_h, d_v)
                if resolved >= reach:
                    continue
                out.append((
                    ((fi.x + fj.x) / 2.0, (fi.y + fj.y) / 2.0),
                    (fi.altitude + fj.altitude) / 2.0,
                    resolved,
                    float(np.clip(1.0 - resolved / reach, 0.0, 1.0)),
                    resolved < 1.0,
                ))
        return out

    # ------------------------------------------------------------------
    # Conflict prediction
    # ------------------------------------------------------------------

    #: Returned by :meth:`predict_conflicts`, one per predicted conflicting pair.
    Conflict = namedtuple(
        "Conflict",
        "i j t_enter t_cpa duration d_cpa d_alt center radius severity")

    def predict_conflicts(self, horizon, flights=None):
        """Conflicts that *will* occur within ``horizon`` seconds if nothing changes.

        ``find_congestion`` answers "who is too close **right now**", which is the
        KPI's question but a poor control signal — by the time it fires the
        conflict has already happened. Absent a clearance every flight holds its
        heading, speed and level, so each track is a straight line and the closest
        approach of a pair has a closed form. No rollout is needed.

        For a pair, let ``r0`` be the relative position and ``dv`` the relative
        velocity. Horizontal separation squared is the quadratic
        ``|r0 + dv t|^2``; setting it equal to ``hotspot_limit^2`` gives the
        interval the pair spends inside the conflict zone, and its vertex gives
        the closest point of approach.

        Vertical separation is constant without a clearance, so a pair already
        stacked beyond ``ALT_MIN_SEP`` can be skipped outright.

        Returns
        -------
        list of :class:`Conflict`, each carrying

        ``t_enter``   seconds until the pair breaches the zone (0 if already in it)
        ``t_cpa``     seconds until closest approach
        ``duration``  seconds spent inside the zone, clipped to the horizon
        ``d_cpa``     horizontal separation at closest approach
        ``d_alt``     vertical separation (constant)
        ``center``    (x, y) midpoint of the pair at closest approach
        ``radius``    the conflict-zone radius, i.e. ``hotspot_limit``
        ``severity``  in (0, 1], ``exp(-d_cpa/radius) * exp(-d_alt/ALT_MIN_SEP)``

        Severity decays exponentially in *both* separations, so a pair that will
        pass head-on at co-altitude scores near 1 while one that merely clips the
        zone boundary scores near ``e^-1``. That ordering is what lets a controller
        triage: not every predicted conflict deserves the same clearance.
        """
        flights = self.flights if flights is None else flights
        out = []
        n = len(flights)
        R = self.hotspot_limit

        for i in range(n):
            fi = flights[i]
            uix, uiy = fi.get_heading_unit_vectors()
            vix, viy = fi.velocity * uix, fi.velocity * uiy
            for j in range(i + 1, n):
                fj = flights[j]
                d_alt = abs(fi.altitude - fj.altitude)
                if d_alt >= self.ALT_MIN_SEP:
                    continue                       # stacked, and stays stacked

                ujx, ujy = fj.get_heading_unit_vectors()
                rx, ry = fj.x - fi.x, fj.y - fi.y
                dvx = fj.velocity * ujx - vix
                dvy = fj.velocity * ujy - viy

                a = dvx * dvx + dvy * dvy
                b = 2.0 * (rx * dvx + ry * dvy)
                c = rx * rx + ry * ry - R * R

                if a < 1e-12:                      # parallel: separation is constant
                    if c > 0.0:
                        continue
                    t_enter, t_exit, t_cpa = 0.0, horizon, 0.0
                else:
                    disc = b * b - 4.0 * a * c
                    if disc <= 0.0:
                        continue                   # never closes to within R
                    sq = math.sqrt(disc)
                    t_enter = max((-b - sq) / (2.0 * a), 0.0)
                    t_exit = min((-b + sq) / (2.0 * a), horizon)
                    if t_exit <= t_enter:
                        continue                   # conflict is outside the horizon
                    # Vertex of the quadratic, clipped into the conflicting window.
                    t_cpa = min(max(-b / (2.0 * a), t_enter), t_exit)

                d_cpa = math.hypot(rx + dvx * t_cpa, ry + dvy * t_cpa)
                cx = fi.x + vix * t_cpa + (rx + dvx * t_cpa) / 2.0
                cy = fi.y + viy * t_cpa + (ry + dvy * t_cpa) / 2.0
                severity = (math.exp(-d_cpa / R)
                            * math.exp(-d_alt / self.ALT_MIN_SEP))

                out.append(self.Conflict(
                    i=i, j=j, t_enter=t_enter, t_cpa=t_cpa,
                    duration=t_exit - t_enter, d_cpa=d_cpa, d_alt=d_alt,
                    center=(cx, cy), radius=R, severity=severity))
        return out

    # ------------------------------------------------------------------
    # Colour helpers
    # ------------------------------------------------------------------

    def get_alt_color(self, alt):
        alt_norm = np.clip((alt - self.ALT_MIN) / (self.DALT * len(self.allowable_altitudes)), 0, 1)
        return mpl.cm.plasma(alt_norm)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_flight_data(self, save_path):
        data = {
            'n': int(self.n),
            'm': int(self.m),
            'airspace_limit': int(self.airspace_limit),
            'collision_x': float(self.collision_x),
            'collision_y': float(self.collision_y),
            'flights': [
                {'heading':  float(f.heading),
                 'velocity': int(f.velocity),
                 'altitude': int(f.altitude),
                 'x':        float(f.x),
                 'y':        float(f.y),
                 'time':     int(f.time)}
                for f in self.flights
            ],
        }
        with open(save_path, 'w') as fh:
            yaml.dump(data, fh)

    @classmethod
    def from_saved_data(cls, data_yaml_fn):
        with open(data_yaml_fn) as fh:
            data = yaml.safe_load(fh)
        sim = cls(data['n'], data['m'], data['airspace_limit'], init_flights=False)
        sim.collision_x = data['collision_x']
        sim.collision_y = data['collision_y']
        for fi in data['flights']:
            sim.flights.append(Flight(
                heading=fi['heading'], velocity=fi['velocity'], altitude=fi['altitude'],
                x=fi['x'], y=fi['y'], time=fi['time'],
            ))
        return sim

    # ------------------------------------------------------------------
    # Animation (requires ffmpeg for mp4 save)
    # ------------------------------------------------------------------

    def animate(self, save_path):
        fig, ax = plt.subplots()
        norm = mpl.colors.Normalize(vmin=self.allowable_altitudes[0],
                                    vmax=self.allowable_altitudes[-1])
        cmap = mpl.cm.plasma
        sm   = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, ticks=self.allowable_altitudes, label='Altitude')

        ax.set_xlim(-self.airspace_limit, self.airspace_limit)
        ax.set_ylim(-self.airspace_limit, self.airspace_limit)
        ax.add_patch(Rectangle((-self.hotspot_limit, -self.hotspot_limit),
                                2 * self.hotspot_limit, 2 * self.hotspot_limit, fill=False))

        arrow_size = 250
        arrows   = []
        fl_texts = []
        for i, flight in enumerate(self.flights):
            c = self.get_alt_color(flight.altitude)
            x, y, *_ = flight.get_5_tuple()
            arrows.append(mpl.patches.FancyArrow(x, y, 0, 0, edgecolor=c, facecolor=c,
                                                  head_width=arrow_size, head_length=arrow_size, zorder=15))
            fl_texts.append(ax.text(0, 0, '', fontsize=6))

        def init():
            for i, flight in enumerate(self.flights):
                dx, dy = flight.get_heading_unit_vectors()
                arrows[i] = mpl.patches.FancyArrow(flight.x, flight.y, dx, dy, width=0.01, color='b')
                ax.add_patch(arrows[i])
                fl_texts[i].set_text('')
            return arrows + fl_texts

        def update(frame):
            for i, flight in enumerate(self.flights):
                was_in = self.flight_within_limit(flight)
                flight.advance_clock(1)
                now_in = self.flight_within_limit(flight)
                if not now_in:
                    continue
                x, y, alt, vel, _ = flight.get_5_tuple()
                cong_cen, cong_alt = self.find_congestion(self.flights)
                for cen, calt in zip(cong_cen, cong_alt):
                    ax.add_patch(plt.Circle(cen, self.hotspot_limit,
                                            color=self.get_alt_color(calt), fill=False, alpha=0.1))
                ux, uy = flight.get_heading_unit_vectors()
                c = self.get_alt_color(alt)
                arrows[i].remove()
                arrows[i] = mpl.patches.FancyArrow(x, y, ux * arrow_size, uy * arrow_size,
                                                    edgecolor=c, facecolor=c,
                                                    head_width=arrow_size, head_length=arrow_size, zorder=15)
                ax.add_patch(arrows[i])
                fl_texts[i].set_position((x, y))
                fl_texts[i].set_text(f'FL{alt:.0f}\nV{vel:.0f}')
                ax.set_title(f'Step={frame}  Congestion events={len(cong_cen)}')
            return arrows + fl_texts

        ani = animation.FuncAnimation(fig, update, frames=50, init_func=init, blit=False, repeat=False)
        ani.save(save_path, dpi=100, writer=mpl.animation.FFMpegWriter(fps=5))
        plt.close(fig)

    def __repr__(self):
        flights_str = ''.join(f'  {i}: {f}\n' for i, f in enumerate(self.flights))
        return (f"CollisionCourseSimulator(\n"
                f"  collision=({self.collision_x:.0f}, {self.collision_y:.0f}),\n"
                f"  n={self.n}, m={self.m}, airspace={self.airspace_limit}\n"
                f"{flights_str})")
