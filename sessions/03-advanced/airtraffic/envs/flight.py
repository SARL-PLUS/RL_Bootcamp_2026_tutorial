import math
import random


class Flight:
    TUPLE5KEYS = ('x', 'y', 'altitude', 'velocity', 'heading')

    def __init__(self, heading=None, velocity=None, altitude=None, x=0., y=0., time=0,
                 VEL_MIN=100, ALT_MIN=100):
        self.VEL_MIN = VEL_MIN
        self.ALT_MIN = ALT_MIN
        self.heading  = heading  if heading  is not None else random.uniform(0, 360)
        self.velocity = velocity if velocity is not None else random.uniform(100, 1000)
        self.altitude = altitude if altitude is not None else random.uniform(1000, 10000)
        self.x = x
        self.y = y
        self.time = time
        self.cur_idx = 0
        self.evo_history = {time: self.get_5_tuple()}
        # These record the value at the time a change was applied (used as a log).
        self.vel_change_history = {}
        self.alt_change_history = {}

    def change_altitude(self, delta):
        self.altitude += delta
        if self.altitude < self.ALT_MIN:
            self.altitude = self.ALT_MIN
        self.alt_change_history[self.time] = self.altitude

    def change_velocity(self, delta):
        self.velocity += delta
        if self.velocity < self.VEL_MIN:
            self.velocity = self.VEL_MIN
        self.vel_change_history[self.time] = self.velocity

    def advance_clock(self, delta_time):
        self.time += delta_time
        distance = self.velocity * delta_time
        ux, uy = self.get_heading_unit_vectors()
        self.x += distance * ux
        self.y += distance * uy
        self.cur_idx += 1
        self.evo_history[self.time] = self.get_5_tuple()

    def reverse_clock(self, delta_time, allow_neg_time=False):
        self.time -= delta_time
        if self.time < 0 and not allow_neg_time:
            self.time = 0
        self.cur_idx = max(0, self.cur_idx - 1)
        distance = self.velocity * delta_time
        ux, uy = self.get_heading_unit_vectors()
        self.x -= distance * ux
        self.y -= distance * uy
        self.evo_history[self.time] = self.get_5_tuple()

    def get_heading_unit_vectors(self):
        rad = math.radians(self.heading)
        return math.cos(rad), math.sin(rad)

    def get_5_tuple(self):
        return self.x, self.y, self.altitude, self.velocity, self.heading

    def get_5_tuple_dict(self):
        return dict(zip(self.TUPLE5KEYS, self.get_5_tuple()))

    def __repr__(self):
        return (f"Flight(heading={self.heading:.2f}, velocity={self.velocity:.2f}, "
                f"altitude={self.altitude:.2f}, x={self.x:.2f}, y={self.y:.2f}, time={self.time:.2f})")
