"""
The improved viewer and video for demo.py (from the UI_Improvements branch), kept in its own file so that demo.py's
code is unchanged. demo.py runs install() from a clearly marked block at its end. install() then:

  * takes the viewer's own flags off the command line (see UI_FLAGS below), so demo.py's own flags work as before,
  * swaps demo.py's show_episodes and record_episode for the ones here (the evaluation and CSV files are untouched),
  * draws the scene with a visual-only copy of Gym/model/leg.xml (sky, floor, lights, finer ground cells) and a smoother
    drawing of the dented ground. Neither changes the physics: the copy only adds or changes geoms that never collide,
    textures and lights, and GroundModel.update_visual is used for drawing only.

Run `python demo.py --classic-viewer` to get the original viewer back. See ../UI_IMPROVEMENTS.md for the full list.
Drafted with Claude Code (Anthropic), for the AI declaration.
"""
import argparse
import contextlib
import json
import sys
import tempfile
import textwrap
import time
import traceback
from pathlib import Path

import numpy as np
import mujoco

from common.env_setup import make_env, to_norm_action, to_physical
from common.evaluate import REFERENCE
from common.policies import FixedPolicy
import stance_env.ankle_gym_env as ankle_gym_env
import stance_env.ground as ground_module
from stance_env.ankle_gym_env import (FRAME_SKIP, MAX_EPISODE_LENGTH, FALL_ANGLE_BACK, FALL_ANGLE_FWD,
                                      COLLAPSE_FRACTION, LAMBDA_BW, K_RANGE, B_RANGE, ANKLE_RANGE, TAU_LIMIT)


# ---------------------------------------------------------------------------------------------------------------------
# The visual-only model: Gym/model/leg.xml with a sky, a floor texture, shadows, a fill light, and 280 ground cells of
# 5 mm (14 cm wide) instead of 70 of 2 cm. Built at run time from the real file, so it follows any change to it.
# ---------------------------------------------------------------------------------------------------------------------
_VISUAL_EDITS = [
    ('    <compiler angle="radian"/>\n', '''    <compiler angle="radian"/>

    <visual>
        <headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0 0 0"/>
        <quality shadowsize="4096" offsamples="8"/>
        <global offwidth="1280" offheight="720"/>
        <map znear="0.01"/>
    </visual>

    <asset>
        <texture name="sky" type="skybox" builtin="gradient"
                 rgb1="0.45 0.65 0.9" rgb2="0.95 0.97 1.0" width="512" height="3072"/>
        <texture name="floor_tex" type="2d" builtin="checker" mark="edge"
                 rgb1="0.80 0.82 0.85" rgb2="0.68 0.70 0.74" markrgb="0.55 0.57 0.6"
                 width="300" height="300"/>
        <material name="floor_mat" texture="floor_tex" texrepeat="10 10" rgba="1 1 1 0.2"/>
    </asset>
'''),
    ('            diffuse="0.9 0.9 0.9"/>\n', '''            diffuse="0.9 0.9 0.9"/>

        <light
            name="fill_light"
            pos="1.5 2 2"
            dir="-0.5 -1 -1"
            diffuse="0.35 0.35 0.4"
            castshadow="false"/>
'''),
    ('            rgba="0.8 0.8 0.8 0.3"/>', '            material="floor_mat"/>'),
    ('<replicate count="70" offset="0.02 0 0">', '<replicate count="280" offset="0.005 0 0">'),
    ('size="0.01 0.15 0.2"\n                pos="-0.39 0 -0.2"', 'size="0.0025 0.07 0.2"\n                pos="-0.3975 0 -0.2"'),
]
_UI_MODEL = None


def _ui_model_path():
    """Write the visual-only copy of leg.xml once (to a temporary folder) and return its path. If leg.xml has changed
    so that an edit no longer applies, that edit is skipped and the scene just looks plainer."""
    global _UI_MODEL
    if _UI_MODEL is None:
        text = Path(ankle_gym_env.MODEL_PATH).read_text()
        for old, new in _VISUAL_EDITS:
            if old in text:
                text = text.replace(old, new, 1)
            else:
                print(f"viewer: leg.xml has changed, one visual edit skipped ({old.strip()[:40]}...)")
        path = Path(tempfile.gettempdir()) / "stance_leg_ui.xml"
        path.write_text(text)
        _UI_MODEL = path
    return _UI_MODEL


@contextlib.contextmanager
def ui_model():
    """Environments made inside this block load the visual-only copy of leg.xml (AnkleEnv reads MODEL_PATH when it is
    made). Used only for the viewer and the video, never for the evaluation."""
    saved = ankle_gym_env.MODEL_PATH
    ankle_gym_env.MODEL_PATH = _ui_model_path()
    try:
        yield
    finally:
        ankle_gym_env.MODEL_PATH = saved


def _blur(values, sigma):
    """Smooth a 1-D profile with a Gaussian of `sigma` samples, keeping the edge values flat."""
    if sigma < 0.5:
        return values
    r = int(3 * sigma) + 1
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    return np.convolve(np.pad(values, r, mode="edge"), k / k.sum(), mode="valid")


def _update_visual(self, model):
    """GroundModel.update_visual with softer dents: sloping trench edges and a low rim of displaced ground. The same as
    the original apart from the marked lines. Drawing only, the forces never see it."""
    if self._cells is None:
        self._cells = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("ground_cell")]
    if not self._cells:
        return
    cell_x = model.geom_pos[self._cells, 0]
    if self._cell_dent is None:
        self._cell_dent = np.zeros(len(self._cells))
    order = np.argsort(self.last_x)
    xs = self.last_x[order]
    under = (cell_x >= xs[0]) & (cell_x <= xs[-1])
    dent_now = np.interp(cell_x, xs, self.surface_z[order])
    sink_now = np.interp(cell_x, xs, np.minimum(self.surface_z, self.last_z)[order])
    self._cell_dent[under] = np.minimum(self._cell_dent[under], dent_now[under])
    top = np.where(under, np.minimum(self._cell_dent, sink_now), self._cell_dent)
    # --- softer look (new) ---
    dx = float(np.median(np.diff(cell_x))) if len(cell_x) > 1 else 0.005
    soft = _blur(top, 0.012 / dx)
    rim = np.clip(_blur(top, 0.015 / dx) - _blur(top, 0.045 / dx), 0.0, None) * 0.6
    rim = np.where(under | (self._cell_dent < -1e-6), 0.0, rim)
    top = np.where(under, top, np.minimum(soft, 0.0) + rim)
    # --- end of softer look ---
    model.geom_pos[self._cells, 2] = top - model.geom_size[self._cells, 2]


# --- viewer start ---
# Everything below is for looking at the agent: a live MuJoCo viewer with a HUD, force arrows,
# a ground coloured by stiffness, keyboard controls and a side-by-side compare mode, plus the
# offscreen renderer that --record and the screenshot key share. None of it changes the physics.
CAMERAS = [  # name, body to follow ("body" or "foot"), distance, azimuth, elevation, height looked at (m)
    ("side", "body", 2.2, 90.0, -8.0, 0.45),
    ("3/4 view", "body", 2.4, 55.0, -18.0, 0.45),
    ("front", "body", 2.4, 0.0, -8.0, 0.45),
    ("foot close-up", "foot", 0.9, 90.0, -6.0, 0.10),
]
HELP = ("ENTER  play / continue / reset (nothing plays on its own)\nSPACE  pause while playing\nLEFT / RIGHT  previous / next episode\n,  /  .  step back / forward one frame (pauses)\nUP / DOWN  faster / slower\n"
        "R  restart episode\nV  change camera\nG  ghost reference leg (compare)\nT  charts, gauges and scorecard\nA  auto-pause at events\nP  save screenshot\nH  hide / show all panels\nI  hide / show the ankle's motor box and the motor-model note\nL  hide / show the legend\nF1  hide / show this list\n"
        "F11  full screen / window    ESC  quit    mouse: drag to orbit, scroll to zoom")
SOFT_COLOUR, HARD_COLOUR = np.array([0.86, 0.66, 0.36]), np.array([0.40, 0.46, 0.58])   # sand -> slate
FELL_COLOUR = np.array([0.90, 0.15, 0.15])
# How good a state is, for the colour of the inference lines.
GOOD, WARN, BAD = "good", "warn", "bad"
LEVEL_COLOUR = {GOOD: (0.25, 1.0, 0.40), WARN: (1.0, 0.92, 0.15), BAD: (1.0, 0.55, 0.05), None: (1.0, 1.0, 1.0),
                "info": (0.55, 0.85, 1.0), "dim": (0.78, 0.82, 0.87), "dimtitle": (0.55, 0.75, 0.92)}
ARROW_Y, ARROW_M_PER_BW, ARROW_MAX = -0.085, 0.35, 0.8     # arrows: y position, metres per body weight (exaggerated), longest
# The ankle motor, drawn on the leg: a teal cylinder at the ankle joint, a ring that grows with stiffness, and a
# cyan arc showing which way (and how hard) it is turning. Drawing only: nothing here is part of the physics.
MOTOR_RGBA, TORQUE_RGBA = (0.0, 0.72, 0.78, 1.0), (0.45, 1.0, 1.0, 0.95)
MOTOR_R, MOTOR_HL, ARC_R = 0.05, 0.055, 0.034                    # motor radius, half length, radius of the torque arc (m): inside the teal disc
ARC_WIDTH, ARC_Y = 0.0045, -0.067                                 # thickness of the arc; its y, just in front of the motor's face (-0.055)
RING_SOFT, RING_STIFFENING, RING_SOFTENING = (0.35, 0.50, 0.55, 1.0), (0.95, 0.25, 0.90, 1.0), (0.30, 0.40, 0.95, 1.0)   # grey, magenta, blue
EVENT_HOLD = 10                                                   # steps an event chip (TOUCHDOWN, PEAK LOAD, FELL) stays up
TAG = {None: "", "good": "[OK]", "warn": "[~]", "bad": "[!]"}   # a symbol for each colour, for anyone who cannot rely on colour
TARGET_RGBA, ACTUAL_RGBA = (1.0, 1.0, 1.0, 1.0), (0.04, 0.08, 0.25, 1.0)   # the motor's target-angle needle and the ankle's actual angle
STATE_HOLD = 6                                                    # steps a "stiffening"/"softening" signal stays up
GHOST_RGBA = (0.95, 0.97, 1.0, 0.28)                                      # the longest an arrow can get
ARROW_WIDTH = 0.006                                                   # arrow thickness; the ghost uses the same width
ARROW_SHAFT = 1.0                                                     # radius of the plain shaft vs the arrow part's
ARROW_HEAD = 37.5 * ARROW_WIDTH                                       # length of the arrow part at the tip (m); keeps the head's shape
TOE_RGBA, HEEL_RGBA = (0.95, 0.05, 0.05, 1.0), (0.62, 0.2, 1.0, 1.0)      # red toe, violet heel (orange is kept for "needs to change")
LEGEND = [(TOE_RGBA[:3], "Toe ground force (arrow)"), (HEEL_RGBA[:3], "Heel ground force (arrow)"),
          (tuple(SOFT_COLOUR), "Soft ground"), (tuple(HARD_COLOUR), "Hard ground"),
          (tuple(FELL_COLOUR), "Leg turns red = fell"),
          (LEVEL_COLOUR[GOOD], "Green [OK]: acceptable"), (LEVEL_COLOUR[WARN], "Yellow [~]: improving, not ideal"),
          (LEVEL_COLOUR[BAD], "Orange [!]: needs to change"),
          (GHOST_RGBA[:3], f"Ghost arrow = max ({ARROW_MAX / ARROW_M_PER_BW:.1f} BW)"),
          (MOTOR_RGBA[:3], "Ankle motor (teal)"), (TORQUE_RGBA[:3], "Cyan arc = motor torque vector"),
          (RING_STIFFENING[:3], "Magenta ring = stiffening up"), (RING_SOFTENING[:3], "Blue ring = softening"),
          (TARGET_RGBA[:3], "White needle = target angle"), ((0.15, 0.25, 0.65), "Dark needle = ankle angle")]
LEGEND_NOTES = ["Arrow length = force, exaggerated", "BW = body weights"]
GHOST_LEG_RGBA = (0.95, 0.97, 1.0, 0.30)                              # the reference leg, drawn over a network's leg in compare mode


def legend_items(compare):
    """The legend rows (colour or None, text). Compare mode adds the ghost leg."""
    return LEGEND + ([(GHOST_LEG_RGBA[:3], "Ghost leg = reference")] if compare else []) + [(None, t) for t in LEGEND_NOTES]
KEY_CAPS = [("ENTER", "play"), ("SPACE", "pause"), ("<- ->", "episode"), (", .", "step"), ("UP/DN", "speed"), ("R", "restart"), ("V", "camera"),
            ("P", "photo"), ("T", "charts"), ("F1", "all keys"), ("ESC", "quit"), ("G", "ghost"), ("A", "auto-pause"),
            ("H/I/L", "hide"), ("F11", "full")]
LEG_GEOMS = ["foot_geom", "shank_geom", "pylon_geom", "upper_leg_geom", "body_mass_geom"]


class Player:
    """One controller playing one episode at a time, plus what the HUD needs to know about it."""

    def __init__(self, label, wrappers, net=None, model=None, vn=None, fixed=None, ground="train"):
        self.label, self.is_fixed = label, bool(fixed)
        if fixed:
            self.brain = "None: a fixed rule, it never reacts"
        elif "lstm" in net.NAME.lower():
            self.brain = "Sees one moment at a time, but remembers what came before"
        else:
            frames = getattr(vn, "frames", 1)
            self.brain = f"No memory, but looks back over the last {frames} moments" if frames > 1 else "Sees only the current moment, no memory"
        with ui_model():                                                  # the visual-only version of leg.xml
            self.env = make_env(ground, wrappers)
        self.policy = FixedPolicy(to_norm_action(self.env, *REFERENCE)) if fixed else net.POLICY(model, vn)
        self.sim = self.env.unwrapped                                    # the AnkleEnv inside the wrappers
        m = self.sim.model
        self.cells = [i for i in range(m.ngeom) if (m.geom(i).name or "").startswith("ground_cell")]
        self.leg = [m.geom(n).id for n in LEG_GEOMS]
        self.leg_rgba = m.geom_rgba[self.leg].copy()                     # to undo the fall tint
        self.ankle_jid = m.joint("ankle_hinge").id
        self.shank_bid = self.sim.shank_bid
        self.stiffen_hold = self.soften_hold = 0                         # steps left on the stiffening/softening signal
        self.vis = mujoco.MjData(m)                                      # what is drawn: the state between two steps
        self.snap_prev = self.snap_cur = None
        self.settled = False

    def reset(self, seed):
        self.seed = seed
        self.obs, _ = self.env.reset(seed=seed)
        self.policy.reset()
        self.ret, self.toe, self.steps = 0.0, 0.0, 0
        self.done, self.fell, self.cause, self.info = False, False, "", {}
        self.prev_phys = self.cur_phys = None                            # last two actions, in real units
        self.force, self.prev_force, self.peak_force = 0.0, 0.0, 0.0
        self.stiffen_hold = self.soften_hold = 0
        self.toe_hist, self.heel_hist, self.tau_hist = [], [], []          # per step: toe force (BW), heel force (BW), motor torque (N.m)
        self.max_tilt, self.min_height = 0.0, 1.0                           # the worst lean and the lowest the body got
        self.touch_step, self.peak_step, self.peak_fired = None, 0, False
        self.event, self.event_hold, self.new_event = None, 0, None         # the latest event, how long to show it, and an unseen one
        self.dent, self.prev_dent = 0.0, 0.0                             # how deep the ground is dented, in metres
        m, g = self.sim.model, self.sim.ground
        k0 = float(g.k0[0])                                              # colour the ground by its stiffness
        t = np.clip((np.log10(k0) - np.log10(2e3)) / (np.log10(2e5) - np.log10(2e3)), 0.0, 1.0)
        self.ground_colour = (1 - t) * SOFT_COLOUR + t * HARD_COLOUR
        if self.cells:
            m.geom_rgba[self.cells, :3] = self.ground_colour
        m.geom_rgba[self.leg] = self.leg_rgba
        g.update_visual(m)
        self.settled = False
        self.snap_prev = self.snap_cur = self._snap(fresh=True)
        self.pose(1.0)

    def _snap(self, fresh=False):
        """The things that change while drawing: joint positions, the ground cells and the two ground forces."""
        sim = self.sim
        toe = 0.0 if fresh else float(sim.toe_force)
        heel = 0.0 if fresh else max(float(sim.ground.last_normal) - toe, 0.0)
        cells = sim.model.geom_pos[self.cells, 2].copy()
        tau = 0.0 if fresh else float(sim.data.ctrl[0])                  # motor torque this step
        k = K_RANGE[0] if fresh else float(self.cur_phys[1])
        theta = 0.0 if fresh else float(self.cur_phys[0])                # target angle, degrees
        return sim.data.qpos.copy(), cells, toe, heel, tau, k, theta

    def pose(self, alpha):
        """Set what is drawn to a point between the previous step (alpha 0) and the current one (alpha 1).
        Every joint is a slide or a hinge, so a straight blend of the positions is exact enough to look smooth.
        The physics itself never sees this."""
        a, b = self.snap_prev, self.snap_cur
        m = self.sim.model
        self.vis.qpos[:] = (1 - alpha) * a[0] + alpha * b[0]
        mujoco.mj_kinematics(m, self.vis)
        mujoco.mj_camlight(m, self.vis)
        if self.cells:
            m.geom_pos[self.cells, 2] = (1 - alpha) * a[1] + alpha * b[1]
        self.toe_vis = (1 - alpha) * a[2] + alpha * b[2]
        self.heel_vis = (1 - alpha) * a[3] + alpha * b[3]
        self.tau_vis = (1 - alpha) * a[4] + alpha * b[4]
        self.k_vis = float(np.exp((1 - alpha) * np.log(a[5]) + alpha * np.log(b[5])))
        self.theta_vis = (1 - alpha) * a[6] + alpha * b[6]

    def step(self):
        self.obs, r, term, trunc, self.info = self.env.step(self.policy.act(self.obs))
        self.ret += r
        self.steps += 1
        self.toe = max(self.toe, self.info["toe_force_bw"])
        self.snap_prev = self.snap_cur
        self.prev_phys, self.cur_phys = self.cur_phys, to_physical(self.sim, self.sim.prev_action)
        self.stiffen_hold, self.soften_hold = max(0, self.stiffen_hold - 1), max(0, self.soften_hold - 1)
        if self.prev_phys is not None:                                   # did it just stiffen up, or soften?
            change = self.cur_phys[1] / self.prev_phys[1] - 1
            if change > 0.03:
                self.stiffen_hold, self.soften_hold = STATE_HOLD, 0
            elif change < -0.03:
                self.soften_hold, self.stiffen_hold = STATE_HOLD, 0
        self.prev_force, self.force = self.force, self.info["normal_force_bw"]
        self.peak_force = max(self.peak_force, self.force)
        self.prev_dent, self.dent = self.dent, float(-self.sim.ground.surface_z.min())
        self.done, self.fell = bool(term or trunc), bool(term)
        self.sim.ground.update_visual(self.sim.model)                    # sink and dent the ground under the foot
        if self.cells:                                                   # dented ground gets darker, like pressed soil
            m = self.sim.model
            depth = -(m.geom_pos[self.cells, 2] + m.geom_size[self.cells, 2])
            m.geom_rgba[self.cells, :3] = self.ground_colour * (1 - 0.4 * np.clip(depth / 0.015, 0, 1))[:, None]
        if self.fell:
            sim = self.sim
            tilt_ok = -FALL_ANGLE_BACK <= sim._leg_tilt() <= FALL_ANGLE_FWD
            self.cause = "collapsed" if tilt_ok else "tipped over"
            sim.model.geom_rgba[self.leg, :3] = FELL_COLOUR              # the leg turns red when it falls
        self.snap_cur = self._snap()
        self._track()

    def _raise(self, text):
        self.event, self.event_hold, self.new_event = text, EVENT_HOLD, text

    def _track(self):
        """Per-step records for the charts, the scorecard and the event markers."""
        sim, bw = self.sim, self.sim.body_weight
        self.toe_hist.append(self.snap_cur[2] / bw)
        self.heel_hist.append(self.snap_cur[3] / bw)
        self.tau_hist.append(self.snap_cur[4])
        tilt = self.info["leg_tilt_deg"]
        if abs(tilt) > abs(self.max_tilt):
            self.max_tilt = tilt
        self.min_height = min(self.min_height, self.info["body_height"] / sim.standing_height)
        self.event_hold = max(0, self.event_hold - 1)
        if not self.event_hold:
            self.event = None
        if self.force >= self.peak_force:
            self.peak_step = self.steps
        if self.fell:
            self._raise("FELL")
        elif not self.peak_fired and self.peak_force > 1.0 and self.force < 0.9 * self.peak_force:
            self.peak_fired = True                                          # the load has peaked and is coming down
            self._raise(f"PEAK LOAD {self.peak_force:.1f} BW")
        elif self.touch_step is None and self.force > 0.05:
            self.touch_step = self.steps
            self._raise("TOUCHDOWN")

    def peak_force_toe(self):
        return self.toe

    def status(self):
        if not self.done:
            return "running"
        return f"FELL ({self.cause})" if self.fell else "survived"

    def rows(self, episode_text):
        """(label, value) rows for the HUD, shared by the viewer and the offscreen renderer."""
        sim = self.sim
        theta, K, B = to_physical(sim, sim.prev_action) if self.steps else (0.0, 0.0, 0.0)
        g = sim.ground
        return [("Controller", self.label), ("Episode", episode_text), ("Step", f"{self.steps}/{MAX_EPISODE_LENGTH}"),
                ("Sim time", f"{self.steps * sim.model.opt.timestep * FRAME_SKIP:.3f} / {MAX_EPISODE_LENGTH * sim.model.opt.timestep * FRAME_SKIP:.3f} s"),
                ("Return", f"{self.ret:.2f}"), ("Toe force", f"{self.info.get('toe_force_bw', 0.0):.2f} BW"),
                ("Peak toe", f"{self.toe:.2f} BW"), ("Ankle target", f"{theta:+.1f} deg"),
                ("Stiffness K", f"{K:.0f} N.m/rad"), ("Damping B", f"{B:.2f} N.m.s/rad"),
                ("Ground k0", f"{g.k0[0] / 1000:.1f} kN/m"), ("Dent", f"{-g.surface_z.min() * 1000:.1f} mm"),
                ("Status", self.status())]

    def motor_effort(self):
        """How hard the ankle motor is working, 0 to 1 of its torque limit, and a word for it."""
        e = abs(float(self.sim.data.ctrl[0])) / TAU_LIMIT
        return e, ("barely working" if e < 0.15 else "pushing steadily" if e < 0.5 else "pushing hard" if e < 0.9 else "at its limit")

    def inference(self):
        """A plain-words reading of this moment from the ankle's point of view. The ankle is the only motor: every
        step it is told a target angle, a stiffness and a damping, and it turns those into torque against the ground
        that is giving way under the foot. Rule-based, from the measured tilt, forces and the chosen action (not the
        network's own reasoning), and free of numbers so it can be read at a glance."""
        if not self.steps:
            return [("Ground", "Foot has not landed yet", GOOD), ("Ankle feels", "-", None), ("Ankle's motor does", "-", None),
                    ("Setting", "-", None), ("Decision", "-", None)]
        sim, info = self.sim, self.info
        theta, K, B = self.cur_phys
        tilt, rate = info["leg_tilt_deg"], float(np.rad2deg(sim._leg_tilt_rate()))
        height = info["body_height"] / sim.standing_height
        toe, f = info["toe_force_bw"], self.force
        fwd, back = np.rad2deg(FALL_ANGLE_FWD), np.rad2deg(FALL_ANGLE_BACK)
        q = float(np.rad2deg(sim.data.qpos[sim.IANKLE]))
        span = np.rad2deg(ANKLE_RANGE[1] - ANKLE_RANGE[0])
        off = abs(q - theta) / span                                      # how far the ankle is from where it was told to be
        sink = (self.dent - self.prev_dent) * 1000                       # mm of new sinking this step
        effort, effort_word = self.motor_effort()

        # What the ankle feels from the ground.
        if f < 0.05:
            ground, g_lvl = "Not touching yet", GOOD
        else:
            giving = "Ground is giving way quickly" if sink > 3 else "Ground is giving way" if sink > 0.5 else \
                     "Ground has dented but is holding" if self.dent > 0.01 else "Ground is holding firm"
            load = "carrying the full weight" if (f >= 0.9 * self.peak_force and self.peak_force > 0.5) else \
                   "weight arriving" if f > self.prev_force + 0.05 else "load easing off"
            ground = f"{giving}, {load}"
            g_lvl = BAD if sink > 3 else WARN if (sink > 0.5 or self.dent > 0.01) else GOOD

        # What the ankle is being asked to hold.
        if self.fell:
            feels, f_lvl = f"Could not hold: the leg {self.cause}", BAD
        else:
            lean = ("leaning forward" if tilt > 0.25 * fwd else "leaning backward" if tilt < -0.25 * back else "upright")
            badly = " (badly)" if (tilt > 0.7 * fwd or tilt < -0.7 * back) else ""
            where = "far from its target" if off > 0.25 else "a bit off its target" if off > 0.1 else "on its target"
            feels = f"Leg {lean}{badly}; ankle {where}"
            leaning = lean != "upright"
            f_lvl = BAD if (badly or (off > 0.25 and (leaning or sink > 0.5))) else WARN if (leaning or off > 0.1) else GOOD

        # What the ankle is doing about it.
        does = effort_word.capitalize() + ("" if effort < 0.15 else " to pull the ankle toward its target")
        d_lvl = BAD if effort >= 0.9 else WARN if effort >= 0.5 else GOOD
        if toe > LAMBDA_BW:
            does += "; landing is harder than comfortable"
            d_lvl = BAD

        stiff, damp = describe_ankle(K, B)
        word = {"stiff": "Stiff", "medium": "Medium", "soft": "Soft"}[stiff]
        feel = {"heavy damping": "heavily damped", "moderate damping": "moderately damped", "light damping": "lightly damped"}[damp]
        effect = "resists being bent" if stiff == "stiff" else "gives way under load" if stiff == "soft" else "bends a little"
        setting = f"{word} and {feel}: {effect}"
        trouble = (not self.fell) and (tilt > 0.25 * fwd or tilt < -0.25 * back or sink > 0.5 or height < COLLAPSE_FRACTION + 0.12)
        s_lvl = (BAD if stiff == "soft" else WARN if stiff == "medium" else GOOD) if trouble else \
                (WARN if (stiff == "stiff" and toe > LAMBDA_BW) or stiff == "soft" else GOOD)

        # What it decided this step, and why that makes sense given what it feels.
        in_trouble = trouble
        c_lvl = GOOD
        if self.prev_phys is None:
            decision = "First move"
        else:
            pt, pk, pb = self.prev_phys
            if K / pk - 1 > 0.03:
                decision = "Stiffening: bracing against the sink" if in_trouble else "Stiffening up"
                c_lvl = WARN if in_trouble else GOOD                    # right direction, but the trouble is still there
            elif K / pk - 1 < -0.03:
                decision = "Softening: cushioning the landing" if toe > 0.8 * LAMBDA_BW else "Softening: yielding to the ground"
                c_lvl = WARN if toe > 0.8 * LAMBDA_BW else (BAD if in_trouble else GOOD)
            elif B / pb - 1 > 0.03:
                decision = "Adding damping: calming the motion"
                c_lvl = WARN if in_trouble else GOOD
            elif B / pb - 1 < -0.03:
                decision = "Reducing damping: letting the ankle move freely"
                c_lvl = WARN if in_trouble else GOOD
            else:
                decision = "Holding the same setting"
                c_lvl = BAD if in_trouble else GOOD                     # trouble and no change: something has to change
        lim = np.rad2deg(ANKLE_RANGE[1])
        k_pos = np.log(K / K_RANGE[0]) / np.log(K_RANGE[1] / K_RANGE[0])
        if in_trouble and k_pos > 0.97 and abs(theta) > 0.97 * lim:
            decision += " - nothing left to push with"
            c_lvl = BAD
        if self.fell:
            c_lvl = BAD
        return [("Ground", ground, g_lvl), ("Ankle feels", feels, f_lvl), ("Ankle's motor does", does, d_lvl),
                ("Setting", setting, s_lvl), ("Decision", decision, c_lvl)]

    def arrows(self, add):
        """Call add(from_point, to_point, rgba) for the heel and toe ground-force arrows."""
        sim = self.sim
        toe, heel = self.toe_vis, self.heel_vis
        for site in sim.sid:                                              # ghost arrows first: the longest an arrow can be
            p = self.vis.site_xpos[site].copy()
            p[1] = ARROW_Y + 0.02                                         # just behind the real arrow, so it never tints it
            add(p, p + np.array([0.0, 0.0, ARROW_MAX]), GHOST_RGBA)
        for site, force, rgba in ((sim.sid[0], heel, HEEL_RGBA), (sim.sid[1], toe, TOE_RGBA)):
            if force > 5.0:
                p = self.vis.site_xpos[site].copy()
                p[1] = ARROW_Y        # drawn just in front of the ground strip, so the ground cannot hide it
                add(p, p + np.array([0.0, 0.0, min(ARROW_MAX, ARROW_M_PER_BW * force / sim.body_weight)]), rgba)


def describe_ankle(K, B):
    k_pos = np.log(K / K_RANGE[0]) / np.log(K_RANGE[1] / K_RANGE[0])
    b_pos = np.log(B / B_RANGE[0]) / np.log(B_RANGE[1] / B_RANGE[0])
    return ("stiff" if k_pos > 0.66 else "medium" if k_pos > 0.33 else "soft",
            "heavy damping" if b_pos > 0.66 else "moderate damping" if b_pos > 0.33 else "light damping")


# Every text each row of the inference box can show (it must match inference()). The box is sized for the biggest
# of them, so it keeps one size however the text changes: a box that resizes with every sentence looks twitchy.
INFERENCE_TITLE = "WHAT THE ANKLE'S MOTOR IS DOING (rule-based reading)"
ROW_VARIANTS = {
    "Ground": ["Not touching yet", "Foot has not landed yet"] + [
        f"{g}, {l}" for g in ("Ground is giving way quickly", "Ground is giving way", "Ground has dented but is holding",
                              "Ground is holding firm")
        for l in ("carrying the full weight", "weight arriving", "load easing off")],
    "Ankle feels": ["Could not hold: the leg tipped over", "Could not hold: the leg collapsed"] + [
        f"Leg {lean}{badly}; ankle {where}" for lean in ("leaning forward", "leaning backward", "upright")
        for badly in ("", " (badly)") for where in ("far from its target", "a bit off its target", "on its target")],
    "Ankle's motor does": [w.capitalize() + ("" if w == "barely working" else " to pull the ankle toward its target") + hard
                           for w in ("barely working", "pushing steadily", "pushing hard", "at its limit")
                           for hard in ("", "; landing is harder than comfortable")],
    "Setting": [f"{s} and {f}: {e}" for s, e in (("Stiff", "resists being bent"), ("Medium", "bends a little"),
                                                  ("Soft", "gives way under load"))
                for f in ("heavily damped", "moderately damped", "lightly damped")],
    "Decision": [t + stuck for t in ("First move", "Stiffening: bracing against the sink", "Stiffening up",
                                      "Softening: cushioning the landing", "Softening: yielding to the ground",
                                      "Adding damping: calming the motion", "Reducing damping: letting the ankle move freely",
                                      "Holding the same setting")
                 for stuck in ("", " - nothing left to push with")],
}
WORST_ROWS = {head: max(texts, key=len) for head, texts in ROW_VARIANTS.items()}        # the longest of each row


# A short explanation of the motor model, drawn in a box on the left. Facts checked against Gym/stance_env/ankle_gym_env.py
# (torque law, 150 N.m limit, 5 ms step, the 11 sensors), agent/actions.py and agent/train.py (the network only nudges a
# fixed starting setting, by at most 0.3 normalised units = 6 degrees of target angle) and the model in leg.xml.
NOTE_TITLE = "HOW THE ANKLE MOTOR IS MODELLED"
NOTE_PARAGRAPHS = [
    "A virtual spring and damper (the leg's own inertia is the mass). Torque = stiffness x (target angle - ankle angle) "
    "- damping x ankle speed, limited to 150 N.m.",
    "TARGET ANGLE: learned, not calculated. Every 5 ms the network reads 11 sensor values (ankle angle, speed and torque, "
    "ground force, pylon load, shank tilt and its rate, vertical acceleration, its last move) and nudges a fixed starting "
    "setting, moving the target by at most 6 deg either way. It was trained by trial and error (PPO) to stay upright and "
    "land softly.",
]
NOTE_SHORT = [      # the same facts, shorter, for the narrow columns of the compare view
    "A virtual spring and damper: torque = stiffness x (target - angle) - damping x speed, max 150 N.m.",
    "TARGET ANGLE is learned: every 5 ms the network reads 11 sensor values and nudges a fixed starting setting, "
    "moving the target at most 6 deg.",
]


def note_lines(chars, short=False):
    """The motor-model note as wrapped (text, level) lines, `chars` characters wide."""
    lines = [(NOTE_TITLE, "dimtitle")]
    for para in (NOTE_SHORT if short else NOTE_PARAGRAPHS):
        lines += [("", "dim")] + [(t, "dim") for t in textwrap.wrap(para, chars)]
    return lines


def wrap_row(head, text, width, tag=""):
    prefix = f"{tag} " if tag else ""
    return textwrap.wrap(f"{prefix}{head.upper()}: {text}", width, subsequent_indent="    ") or [""]


def inference_lines(player, width):
    """The inference box as (text, level) lines, `width` characters wide, with the colour level of each line.
    Every row keeps the number of lines its longest possible text needs, so the lines never move."""
    lines = [(INFERENCE_TITLE, None)]
    for head, text, level in player.inference():
        wrapped = wrap_row(head, text, width, TAG[level])
        room = max(len(wrap_row(head, t, width, "[OK]")) for t in ROW_VARIANTS[head])      # worst case: the longest tag
        lines += [(t, level) for t in wrapped] + [("", level)] * max(0, room - len(wrapped))
    return lines


def inference_worst(width):
    """The text of the biggest possible inference box, to size the box by."""
    lines = [(INFERENCE_TITLE, None)]
    for head, text in WORST_ROWS.items():
        lines += [(t, None) for t in wrap_row(head, text, width, "[OK]")]
    return lines


def set_camera(cam, sim, index):
    """Put the camera on preset `index`: it follows the body (or foot) along x and looks at a fixed height."""
    name, body, dist, az, el, height = CAMERAS[index % len(CAMERAS)]
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    aim_camera(cam, sim, index)
    return name


def aim_camera(cam, sim, index, data=None):
    _, body, _, _, _, height = CAMERAS[index % len(CAMERAS)]
    xpos = (data or sim.data).xpos
    if body == "foot":                      # the foot close-up also follows the foot down when the ground gives way
        pos = xpos[sim.foot_bid]
        cam.lookat[:] = (pos[0], 0.0, pos[2] + height)
    else:                                   # the others look at a fixed height, halfway between the foot and the body in x,
        mid = 0.5 * (xpos[sim.foot_bid][0] + xpos[sim.body_bid][0])     # so a leg that tips over does not leave the picture
        cam.lookat[:] = (mid, 0.0, height)


def draw_arrows_in(scene, player):
    """Add the force arrows to an mjvScene (the viewer's user scene, or a Renderer's scene)."""
    def add(a, b, rgba, width=ARROW_WIDTH):
        # An arrow geom stretches its head along with its length, so a long arrow gets a long, thin head. Instead,
        # a plain shaft runs most of the way and only the last ARROW_HEAD metres are an arrow, so the head keeps its shape.
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        length = float(np.linalg.norm(b - a))
        start = a
        if length > ARROW_HEAD:
            start = b - (b - a) * (ARROW_HEAD / length)
            if scene.ngeom < scene.maxgeom:
                shaft = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(shaft, mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.zeros(9),
                                    np.array(rgba, dtype=np.float32))
                mujoco.mjv_connector(shaft, mujoco.mjtGeom.mjGEOM_CYLINDER, width * ARROW_SHAFT, a, start)
                scene.ngeom += 1
        if scene.ngeom >= scene.maxgeom:
            return
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                            np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, start, b)
        scene.ngeom += 1
    player.arrows(add)
    draw_motor(scene, player)


def draw_motor(scene, player):
    """The ankle motor on the leg: a teal cylinder at the ankle joint, a ring whose size follows the stiffness (yellow
    while it is stiffening up, blue while it softens), and a cyan arc that points the way the motor is turning and
    grows with how hard it pushes. mjv_initGeom fills in a geom and mjv_connector stretches it between two points."""
    sim = player.sim
    c = player.vis.xanchor[player.ankle_jid].copy()                 # the ankle joint in the world
    face = np.array([0.0, -MOTOR_HL, 0.0])                           # the side facing the camera (the hinge axis is y)
    to_y = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0])    # turns a cylinder's long axis from z to y

    def add(kind, size, pos, mat, rgba):
        if scene.ngeom >= scene.maxgeom:
            return None
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float), np.asarray(mat, dtype=float),
                            np.array(rgba, dtype=np.float32))
        scene.ngeom += 1
        return g

    k_pos = float(np.clip(np.log(player.k_vis / K_RANGE[0]) / np.log(K_RANGE[1] / K_RANGE[0]), 0, 1))
    ring_colour = RING_STIFFENING if player.stiffen_hold else RING_SOFTENING if player.soften_hold else RING_SOFT
    ring_r = MOTOR_R * (1.12 + 0.40 * k_pos) * (1.12 if player.stiffen_hold else 1.0)
    ring_hl = 0.012
    add(mujoco.mjtGeom.mjGEOM_CYLINDER, [ring_r, ring_hl, 0], c + face + np.array([0, ring_hl + 0.004, 0]), to_y, ring_colour)
    add(mujoco.mjtGeom.mjGEOM_CYLINDER, [MOTOR_R, MOTOR_HL, 0], c, to_y, MOTOR_RGBA)

    tau = player.tau_vis
    if abs(tau) > 2.0:                                               # the torque arc, in front of the motor
        span = np.deg2rad(300.0) * min(1.0, abs(tau) / TAU_LIMIT)
        sign = 1.0 if tau > 0 else -1.0                              # + torque turns the joint clockwise as seen from the camera
        centre = np.array([c[0], ARC_Y, c[2]])                       # on the motor's face
        n_seg = max(3, int(np.ceil(span / np.deg2rad(8))))           # fine steps: a smooth arc, not a string of beads
        angles = [-np.pi / 2 + sign * span * j / n_seg for j in range(n_seg + 1)]
        pts = [centre + ARC_R * np.array([np.cos(t), 0.0, -np.sin(t)]) for t in angles]
        head_from = max(0, n_seg - 4)                                # the last ~30 degrees become the arrow
        for j in range(head_from):                                   # the thin arc
            g = add(mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.zeros(9), TORQUE_RGBA)
            if g is not None:                                        # cylinders, not capsules: no round ends, so no beads
                mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, ARC_WIDTH, pts[j], pts[j + 1])
        t_end = angles[-1]                                           # the arrow runs on a little past the end, along the tangent
        tangent = sign * np.array([-np.sin(t_end), 0.0, -np.cos(t_end)])
        head = add(mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), TORQUE_RGBA)
        if head is not None:
            mujoco.mjv_connector(head, mujoco.mjtGeom.mjGEOM_ARROW, ARC_WIDTH * 1.7, pts[head_from], pts[-1] + 0.02 * tangent)

    # Two needles on the motor's face: where the ankle is (dark) and where the motor is trying to put it (white).
    shank_up = player.vis.xmat[player.shank_bid].reshape(3, 3)[:, 2]
    phi = np.deg2rad(player.theta_vis) - float(player.vis.qpos[sim.IANKLE])       # how far the target is from the ankle's angle

    def in_screen(vec, a):                                           # turn a vector by a about the hinge axis (y), keep x and z
        return np.array([vec[0] * np.cos(a) + vec[2] * np.sin(a), 0.0, -vec[0] * np.sin(a) + vec[2] * np.cos(a)])

    for vec, rgba, y in ((in_screen(shank_up, 0.0), ACTUAL_RGBA, -0.071), (in_screen(shank_up, phi), TARGET_RGBA, -0.075)):
        centre = np.array([c[0], y, c[2]])
        tip = centre + 0.042 * vec / max(np.linalg.norm(vec), 1e-9)
        g = add(mujoco.mjtGeom.mjGEOM_CYLINDER, np.zeros(3), np.zeros(3), np.zeros(9), rgba)
        if g is not None:
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CYLINDER, 0.0035, centre, tip)
        if rgba is TARGET_RGBA:                                      # a dot on the target needle's tip
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.007, 0, 0], tip, np.eye(3).flatten(), rgba)


def render_frame(renderer, player, cam, episode_text, hud=True, cam_index=0):
    """One offscreen frame of the player's scene, with the force arrows and the HUD drawn in."""
    from PIL import Image, ImageDraw, ImageFont
    aim_camera(cam, player.sim, cam_index, player.vis)
    renderer.update_scene(player.vis, camera=cam)
    draw_arrows_in(renderer.scene, player)
    img = Image.fromarray(renderer.render())
    if hud:
        font = ImageFont.load_default(size=15)
        rows = player.rows(episode_text)
        d = ImageDraw.Draw(img, "RGBA")
        banner = f"{player.label.upper()}: {player.brain}"            # what kind of network this is, along the top
        d.rectangle([0, 0, 8 + int(font.getlength(banner)) + 16, 26], fill=(10, 14, 24, 190))
        d.text((8, 4), banner, fill=tuple(int(255 * c) for c in LEVEL_COLOUR["info"]), font=font)
        T = 34                                                          # the stats panel starts under the banner
        d.rectangle([8, T, 268, T + 6 + 20 * len(rows)], fill=(0, 0, 0, 140))
        for i, (k, v) in enumerate(rows):
            colour = (255, 110, 110) if (k == "Status" and player.fell) else (255, 255, 255)
            d.text((16, T + 4 + 20 * i), k, fill=(190, 200, 215), font=font)
            d.text((120, T + 4 + 20 * i), v, fill=colour, font=font)
        chip = "STIFFENING" if player.stiffen_hold else "SOFTENING" if player.soften_hold else None
        if chip:                                                        # the same chip as in the viewer
            colour = tuple(int(255 * c) for c in (RING_STIFFENING if player.stiffen_hold else RING_SOFTENING)[:3])
            d.rectangle([280, T, 280 + int(font.getlength(chip)) + 20, T + 24], fill=colour)
            d.text((290, T + 3), chip, fill=(255, 255, 255), font=font)
        if player.event:                                                # touchdown / peak load / fell
            ev_bg = (217, 31, 31) if player.event == "FELL" else (64, 77, 115)
            ex = 280 + (int(font.getlength(chip)) + 36 if chip else 0)
            d.rectangle([ex, T, ex + int(font.getlength(player.event)) + 20, T + 24], fill=ev_bg)
            d.text((ex + 10, T + 3), player.event, fill=(255, 255, 255), font=font)
        w = img.width
        y0 = T + 6 + 20 * len(rows) + 14                                   # the motor-model note: small and dim, like in the viewer
        nfont = ImageFont.load_default(size=12)
        nl = note_lines(44)
        d.rectangle([8, y0 - 6, 330, y0 + 15 * len(nl) + 2], fill=(10, 14, 24, 110))
        for j, (line, level) in enumerate(nl):
            d.text((16, y0 + 15 * j), line, fill=tuple(int(255 * c) for c in LEVEL_COLOUR[level]), font=nfont)
        sc = View.scorecard(None, player)                                  # the scorecard, under the note
        sy = y0 + 15 * len(nl) + 16
        d.rectangle([8, sy - 6, 330, sy + 15 * len(sc) + 2], fill=(10, 14, 24, 150))
        for j, (line, level) in enumerate(sc):
            d.text((16, sy + 15 * j), line, fill=tuple(int(255 * c) for c in LEVEL_COLOUR[level]), font=nfont)
        lines = inference_lines(player, 62)
        top = img.height - 16 - 19 * len(lines)
        d.rectangle([w - 560, top - 6, w - 8, img.height - 8], fill=(0, 0, 0, 150))
        for j, (line, level) in enumerate(lines):
            d.text((w - 552, top + 19 * j), line, fill=tuple(int(255 * c) for c in LEVEL_COLOUR[level]), font=font)
        items = legend_items(False)
        d.rectangle([w - 300, 8, w - 8, 14 + 22 * len(items)], fill=(0, 0, 0, 140))
        for j, (c, text) in enumerate(items):
            if c is not None:
                d.rectangle([w - 292, 14 + 22 * j, w - 278, 28 + 22 * j], fill=tuple(int(255 * x) for x in c))
            d.text((w - 268, 12 + 22 * j), text, fill=(255, 255, 255), font=font)
    return np.asarray(img)


class Controls:
    """What the keyboard changed. The viewer's key callback runs in another thread and only sets these."""

    def __init__(self, slow, help_on=True):
        self.slow, self.paused, self.hud, self.camera, self.help = slow, False, True, 0, help_on
        self.inference, self.legend = True, None                        # legend None = decide from the window size
        self.jump, self.restart, self.screenshot, self.step_req = 0, False, False, 0
        self.enter, self.space = False, False
        self.panel = False                                               # charts, gauges and the scorecard: off until T is pressed
        self.ghost, self.autopause = True, False                          # reference ghost leg; pause at events
        self.user_pause = False                                          # paused by the user (SPACE or a step key), not by a prompt

    def key(self, keycode):
        if keycode == 32:                      # SPACE: pause while playing, otherwise the same as ENTER
            self.space = True
        elif keycode in (257, 335):            # ENTER, keypad ENTER: play / continue / reset
            self.enter = True
        elif keycode == 262:                   # RIGHT
            self.jump = 1
        elif keycode == 263:                   # LEFT
            self.jump = -1
        elif keycode == 265:                   # UP: faster
            self.slow = max(0.5, self.slow / 1.5)
        elif keycode == 264:                   # DOWN: slower
            self.slow = min(200.0, self.slow * 1.5)
        elif keycode == 44:                    # ,  one step back (pauses)
            self.step_req = -1
        elif keycode == 46:                    # .  one step forward (pauses)
            self.step_req = 1
        elif keycode == ord("R"):
            self.restart = True
        elif keycode == ord("V"):
            self.camera += 1
        elif keycode == ord("P"):
            self.screenshot = True
        elif keycode == ord("H"):
            self.hud = not self.hud
        elif keycode == ord("T"):
            self.panel = not self.panel
        elif keycode == ord("G"):
            self.ghost = not self.ghost
        elif keycode == ord("A"):
            self.autopause = not self.autopause
        elif keycode == ord("L"):
            self.legend = not self.legend
        elif keycode == ord("I"):
            self.inference = not self.inference
        elif keycode == 290:                   # F1
            self.help = not self.help


def draw_box(ctx, area, lines, ls, right, top=False, widths=None, alpha=0.80):
    """A dark box of coloured text lines in a corner of `area`: bottom (top if `top`), right-hand if `right`, else left.
    `lines` is a list of (text, level); the level picks the colour (green, yellow or orange, white if None).
    mjr_overlay can only draw white text, so each line is drawn with mjr_text, twice a pixel apart to look bolder."""
    lh, pad = int(23 * ls), int(8 * ls)
    text_w = max(widths) if widths else max(len(t) for t, _ in lines) * 11.4 * ls      # measured, or estimated
    w = int(min(area.width, text_w + 2 * pad + 2))
    h = lh * len(lines) + 2 * pad
    x = area.left + (area.width - w if right else 0)
    mujoco.mjr_rectangle(mujoco.MjrRect(x, area.bottom + (area.height - h if top else 0), w, h), 0.05, 0.06, 0.10, alpha)   # also sets the viewport
    for j, (text, level) in enumerate(lines):
        y = 1 - (pad + lh * (j + 0.5) + 14 * ls) / h                  # mjr_text puts text about 14 px above this y
        for dx in (0, 1):
            mujoco.mjr_text(0, text, ctx, (pad + dx) / w, y, *LEVEL_COLOUR[level])


class View:
    """The viewer window: our own GLFW window, so its size is ours (1920x1080, full screen by default).
    With several players the window is split into one column per player, all drawn from the same seed."""

    def __init__(self, players, ctl, size=(1920, 1080), fullscreen=True):
        import glfw
        self.glfw, self.ctl, self.players, self.size = glfw, ctl, players, size
        if not glfw.init():
            raise RuntimeError("could not start GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        monitor = glfw.get_primary_monitor() if fullscreen else None
        self.window = glfw.create_window(size[0], size[1], "STANCE demo", monitor, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not open a window")
        glfw.make_context_current(self.window)
        for _ in range(30):                                              # let the window reach its real size before the first frame:
            glfw.swap_buffers(self.window)                               # on some desktops it is smaller for the first few frames
            glfw.poll_events()
            time.sleep(0.02)
        glfw.swap_interval(1)
        self.fullscreen, self.saved_pos = fullscreen, (100, 50)
        self.note_ctx = mujoco.MjrContext(players[0].sim.model, 100)      # a smaller font, for the motor-model note
        self.nfs = 100
        self.note_cache = {}
        self.text_widths = {}                                            # measured width in pixels of each line of text
        self.fs = 150                                                    # font scale of the text, follows the window size
        self.pctx = mujoco.MjrContext(players[0].sim.model, 300)         # big text, for the prompt
        self.pfs = 300
        self.ctxs = [mujoco.MjrContext(p.sim.model, self.fs) for p in players]
        self.scns = [mujoco.MjvScene(p.sim.model, maxgeom=2000) for p in players]
        self.cams = [mujoco.MjvCamera() for _ in players]
        self.opt = mujoco.MjvOption()
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = False
        self.drag = None
        self.pert = mujoco.MjvPerturb()
        glfw.set_key_callback(self.window, self._key)
        glfw.set_cursor_pos_callback(self.window, self._move)
        glfw.set_mouse_button_callback(self.window, self._button)
        glfw.set_scroll_callback(self.window, self._scroll)

    # --- input ---
    def _key(self, window, key, scancode, action, mods):
        if action != self.glfw.PRESS:
            return
        if key == self.glfw.KEY_ESCAPE:
            self.glfw.set_window_should_close(window, True)
        elif key == self.glfw.KEY_F11:
            self.toggle_fullscreen()
        else:
            self.ctl.key(key)

    def _button(self, window, button, action, mods):
        g = self.glfw
        self.drag = button if action == g.PRESS else None

    def _move(self, window, x, y):
        if self.drag is not None and self.last is not None:
            h = max(1, self.glfw.get_window_size(window)[1])
            dx, dy = (x - self.last[0]) / h, (y - self.last[1]) / h
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if self.drag == self.glfw.MOUSE_BUTTON_LEFT \
                else mujoco.mjtMouse.mjMOUSE_ZOOM
            for p, scn, cam in zip(self.players, self.scns, self.cams):
                mujoco.mjv_moveCamera(p.sim.model, action, dx, dy, scn, cam)
        self.last = (x, y)

    last = None

    def _scroll(self, window, dx, dy):
        for p, scn, cam in zip(self.players, self.scns, self.cams):
            mujoco.mjv_moveCamera(p.sim.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * dy, scn, cam)

    def toggle_fullscreen(self):
        g = self.glfw
        if self.fullscreen:
            g.set_window_monitor(self.window, None, *self.saved_pos, *self.size, 0)
        else:
            self.saved_pos = g.get_window_pos(self.window)
            mon = g.get_primary_monitor()
            g.set_window_monitor(self.window, mon, 0, 0, *self.size, g.DONT_CARE)
        self.fullscreen = not self.fullscreen

    # --- drawing ---
    def is_running(self):
        return not self.glfw.window_should_close(self.window)

    def set_camera(self, index):
        for p, cam in zip(self.players, self.cams):
            set_camera(cam, p.sim, index)

    def render(self, rows_by_player, box_lines, help_text, status_text, save_to=None, prompt=None):
        g = self.glfw
        FW, FH = g.get_framebuffer_size(self.window)
        W, H = (FW, FW * 9 // 16) if FW * 9 <= FH * 16 else (FH * 16 // 9, FH)    # biggest 16:9 picture that fits
        ox, oy = (FW - W) // 2, (FH - H) // 2                                      # black bars around it
        n = len(self.players)
        fs = 150 if W // n >= 900 else 100                                # text size that fits this window (smaller is unreadable)
        if fs != self.fs:
            for ctx in self.ctxs:
                mujoco.mjr_changeFont(fs, ctx)
            self.fs = fs
        ls = fs / 150                                                      # layout scale for everything in pixels
        # Work out the text first. Measuring a line draws it, so it has to happen before the picture is drawn.
        strip = int(40 * ls)                                               # height of the key strip along the bottom
        banners = [[(t, "info") for t in textwrap.wrap(f"{p.label.upper()}: {p.brain}", int((W // n - 16) / (11.4 * ls)))]
                   for p in self.players]                                  # what kind of network each column is
        bh = int(23 * ls) * max(len(b) for b in banners) + 2 * int(8 * ls) if self.ctl.hud else 0
        boxes = []
        for i in range(n):
            if n == 1:                                                     # one column: a box at the bottom right
                width = int(min(76, (W // n) * 0.5 / (11.4 * ls)))
            else:                                                          # several columns: a box along the bottom of each
                width = int((W // n - 20) / (11.4 * ls))
            boxes.append(box_lines(i, width) if (self.ctl.hud and self.ctl.inference) else [])
            worst_width = width
        worst = inference_worst(worst_width) if (self.ctl.hud and self.ctl.inference) else []
        # The motor-model note sits on the left under the stats panel (first column only), where there is room. It is
        # drawn small and dim so that it stays in the background.
        nfs = 100 if fs >= 100 else 50
        if nfs != self.nfs:
            mujoco.mjr_changeFont(nfs, self.note_ctx)
            self.nfs = nfs
        nls = ls * nfs / fs                                                # layout scale of the note's smaller font
        note_pad = int(8 * nls)
        note_inner = int(46 * 10.6 * nls) if n == 1 else int((W // n) * 0.34) - 2 * note_pad     # width of the note's text, px
        note_layout = None
        if self.ctl.hud and self.ctl.inference:
            note_layout = self.note_layout(NOTE_PARAGRAPHS if n == 1 else NOTE_SHORT, note_inner, FW, FH)
            note_title = NOTE_TITLE if self.word_width(NOTE_TITLE, FW, FH) <= note_inner else "THE MOTOR MODEL"
        stats_h = int(len(rows_by_player[0]) * 34.5 * ls) + int(6 * ls)                       # height of the stats panel
        measured = {}
        if self.ctl.hud:
            for lines in banners + boxes + [worst]:
                for text, _ in lines:
                    if (self.fs, text) not in self.text_widths:
                        self.text_widths[(self.fs, text)] = self.measure(text, FW, FH)
                    measured[text] = self.text_widths[(self.fs, text)]
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, FW, FH), 0, 0, 0, 1)
        for i, (p, ctx, scn, cam) in enumerate(zip(self.players, self.ctxs, self.scns, self.cams)):
            aim_camera(cam, p.sim, self.ctl.camera, p.vis)
            mujoco.mjv_updateScene(p.sim.model, p.vis, self.opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
            if n > 1 and i > 0 and self.ctl.ghost:                       # the reference leg, see-through, over this network's leg
                first = scn.ngeom
                mujoco.mjv_addGeoms(self.players[0].sim.model, self.players[0].vis, self.opt, self.pert,
                                    mujoco.mjtCatBit.mjCAT_DYNAMIC, scn)
                for k in range(first, scn.ngeom):
                    scn.geoms[k].rgba[:] = GHOST_LEG_RGBA
                    scn.geoms[k].size[:] = scn.geoms[k].size * 1.12            # a little bigger: it shows as a shell, no z-fighting
                    scn.geoms[k].matid = -1
                    scn.geoms[k].transparent = 1
            draw_arrows_in(scn, p)
            rect = mujoco.MjrRect(ox + i * W // n, oy, W // n, H)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)
            mujoco.mjr_render(rect, scn, ctx)
            font = mujoco.mjtFontScale.mjFONTSCALE_200 if n == 1 else mujoco.mjtFontScale.mjFONTSCALE_150
            if self.ctl.hud:
                rows = rows_by_player[i]
                draw_box(ctx, rect, banners[i], ls, False, top=True, widths=[measured[t] for t, _ in banners[i]])
                state = "STIFFENING" if p.stiffen_hold else "SOFTENING" if p.soften_hold else None
                if p.event:                                # events: touchdown, peak load, fall (second row, same place)
                    cw, chh = 14.6 * ls, int(34 * ls)
                    ev_w = int((len(p.event) + 3) * cw)
                    ev = mujoco.MjrRect(rect.left + int(345 * ls), rect.bottom + rect.height - bh - 2 * chh - int(12 * ls), ev_w, chh)
                    ev_bg = (0.85, 0.12, 0.12, 1.0) if p.event == "FELL" else (0.25, 0.30, 0.45, 1.0)
                    mujoco.mjr_label(ev, 0, p.event, *ev_bg, 1, 1, 1, ctx)
                    mujoco.mjr_label(mujoco.MjrRect(ev.left + 1, ev.bottom, ev_w, chh), 0, p.event, 0, 0, 0, 0, 1, 1, 1, ctx)
                if state:                                  # a chip under the banner: the same colours as the motor's ring
                    cw, chh = 14.6 * ls, int(34 * ls)
                    cwid = int((len(state) + 3) * cw)
                    chip = mujoco.MjrRect(rect.left + int(345 * ls), rect.bottom + rect.height - bh - chh - int(6 * ls), cwid, chh)
                    bg, fg = (RING_STIFFENING if p.stiffen_hold else RING_SOFTENING), (1, 1, 1)
                    mujoco.mjr_label(chip, 0, state, *bg, *fg, ctx)
                    mujoco.mjr_label(mujoco.MjrRect(chip.left + 1, chip.bottom, cwid, chh), 0, state, 0, 0, 0, 0, *fg, ctx)
                below = mujoco.MjrRect(rect.left, rect.bottom, rect.width, rect.height - bh)          # the stats sit under the banner
                mujoco.mjr_overlay(font, mujoco.mjtGridPos.mjGRID_TOPLEFT, below,
                                   "\n".join(k for k, _ in rows), "\n".join(v for _, v in rows), ctx)
                box = mujoco.MjrRect(rect.left, rect.bottom + strip, rect.width, rect.height - strip)   # above the key strip
                if self.ctl.inference:
                    fixed = max([measured[t] for t, _ in worst] + [measured[t] for t, _ in boxes[i]])   # one size for every frame
                    draw_box(ctx, box, boxes[i], ls, n == 1, widths=[fixed])
                    if self.ctl.panel:
                        self.draw_panel(rect, p, ls, nls, bh, n)
                    if i == 0 and note_layout:             # the note, under the stats panel on the far left
                        below_note = strip + (int(23 * ls) * len(boxes[0]) + 2 * int(8 * ls) if n > 1 else 0)   # the inference box is under it
                        area = mujoco.MjrRect(rect.left, rect.bottom + below_note, rect.width, rect.height - bh - stats_h - below_note - int(8 * ls))
                        if area.height >= int(23 * nls) * (1 + len(note_layout)) + 2 * note_pad:     # only if it fits
                            self.draw_note(area, note_title, note_layout, note_inner, nls, 0.5)
            elif i == 0:
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                                   rect, "H: show panel", "", ctx)
        if self.ctl.hud:                              # the keys, always visible along the bottom
            self.draw_keys(ox, oy, W, strip, ls, status_text)
            if self.ctl.help:                         # the full list, in the middle
                mujoco.mjr_overlay(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                                   mujoco.MjrRect(ox + W // 2 - int(330 * ls), oy + H // 2 - int(170 * ls), int(660 * ls), int(340 * ls)),
                                   help_text, "", self.ctxs[0])
        if prompt:                                    # the big prompt, pulsing so it is hard to miss
            pfs = 300 if ls >= 1 else 200 if ls >= 0.66 else 150
            if pfs != self.pfs:
                mujoco.mjr_changeFont(pfs, self.pctx)
                self.pfs = pfs
            cw = 14.6 * pfs / 150
            pw, ph = int((len(prompt) + 3) * cw), int(cw * 2.8)
            pulse = 0.8 + 0.2 * np.sin(time.time() * 6)
            rect = mujoco.MjrRect(ox + W // 2 - pw // 2, oy + int(H * 0.46), pw, ph)
            mujoco.mjr_label(rect, 0, prompt, 1.0, 0.1, 0.6, pulse, 1, 1, 1, self.pctx)          # hot pink, white text
            mujoco.mjr_label(mujoco.MjrRect(rect.left + 2, rect.bottom, pw, ph), 0, prompt, 0, 0, 0, 0, 1, 1, 1, self.pctx)
            hint = "Press  ,  .  to step forward and backward"             # second line, same colour, smaller
            cw2 = 13.5 * ls
            hw, hh = max(pw, int((len(hint) + 3) * cw2)), int(cw2 * 2.4)
            hrect = mujoco.MjrRect(ox + W // 2 - hw // 2, rect.bottom - hh - int(6 * ls), hw, hh)
            mujoco.mjr_label(hrect, 0, hint, 1.0, 0.1, 0.6, pulse, 1, 1, 1, self.ctxs[0])
            mujoco.mjr_label(mujoco.MjrRect(hrect.left + 1, hrect.bottom, hw, hh), 0, hint, 0, 0, 0, 0, 1, 1, 1, self.ctxs[0])
        if self.ctl.legend is None:                   # small window: start without the legend, it would cover the picture
            self.ctl.legend = H >= 700 and W // n >= 900        # long, so it starts hidden when the columns are narrow
        if self.ctl.hud and self.ctl.legend:          # legend, top right of the picture
            font = mujoco.mjtFontScale.mjFONTSCALE_100
            row, bx, sw = int(30 * ls), ox + W - int(400 * ls), int(20 * ls)
            for j, (colour, text) in enumerate(legend_items(n > 1)):
                y = oy + H - bh - int(12 * ls) - row * (j + 1)
                if colour is not None:
                    mujoco.mjr_rectangle(mujoco.MjrRect(bx, y + int(4 * ls), sw, sw), *colour, 1)
                mujoco.mjr_overlay(font, mujoco.mjtGridPos.mjGRID_TOPLEFT, mujoco.MjrRect(bx + int(28 * ls), y, int(370 * ls), row),
                                   text, "", self.ctxs[0])
        if save_to:                                   # read the finished picture (with the HUD) before the swap
            from PIL import Image
            rgb = np.zeros((H, W, 3), dtype=np.uint8)
            mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(ox, oy, W, H), self.ctxs[0])
            Path(save_to).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.flipud(rgb)).save(save_to)
        g.swap_buffers(self.window)
        g.poll_events()

    def measure(self, text, FW, FH, ctx=None):
        """Width in pixels of `text` in the current font: draw it in a corner and see how far it reaches."""
        ctx = ctx or self.ctxs[0]
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)
        w, h = min(FW, 2400), min(FH, 48)
        mujoco.mjr_rectangle(mujoco.MjrRect(0, 0, w, h), 0, 0, 0, 1)
        mujoco.mjr_text(0, text, ctx, 0.0, 0.3, 1, 1, 1)
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb, None, mujoco.MjrRect(0, 0, w, h), ctx)
        cols = np.nonzero(rgb.max(axis=(0, 2)) > 60)[0]
        return int(cols.max()) + 2 if len(cols) else 0

    # ---- the data panel: two charts, two gauges and a scorecard -------------------------------------------------
    def draw_panel(self, rect, p, ls, nls, bh, n):
        """Ground force and motor torque against time, a lean gauge, a body-height gauge and a scorecard, stacked in
        a column: left of the leg in a single view, right of it in the compare view."""
        ctx = self.note_ctx
        gap = int(6 * ls)
        if n == 1:
            px, pw = rect.left + int(350 * ls), int(420 * ls)
        else:
            px = rect.left + int(0.64 * rect.width)
            pw = rect.left + rect.width - px - int(8 * ls)
        chart_h, gauge_h = max(int(100 * ls), 125), max(int(44 * ls), 40)        # mjr_figure needs room for its title and ticks
        y = rect.bottom + rect.height - bh - int(90 * ls)                      # top of the stack, measured from the bottom
        dt = p.sim.model.opt.timestep * FRAME_SKIP
        now = p.steps * dt
        floor = rect.bottom + int(0.27 * rect.height)                          # keep clear of the foot and the ground
        # 1. ground force (BW): toe red, heel violet, the comfort limit orange, "now" white
        if y - chart_h >= floor:
            y -= chart_h
            self.chart(mujoco.MjrRect(px, y, pw, chart_h), "Ground force (BW)", dt, now,
                       [(p.toe_hist, (1.0, 0.2, 0.2)), (p.heel_hist, (0.7, 0.4, 1.0))],
                       (0.0, max(4.5, 1.1 * max(p.toe_hist + p.heel_hist + [0]))), LAMBDA_BW, p, "%.1f")
        # 2. motor torque (N.m): cyan, "now" white
        if y - gap - chart_h >= floor:
            y -= gap + chart_h
            self.chart(mujoco.MjrRect(px, y, pw, chart_h), "Motor torque (N.m)", dt, now, [(p.tau_hist, (0.4, 1.0, 1.0))],
                       (-TAU_LIMIT, TAU_LIMIT), None, p)
        # 3. lean gauge: how close the leg is to tipping over, back or forward
        tilt = p.info.get("leg_tilt_deg", 0.0)
        back, fwd = np.rad2deg(FALL_ANGLE_BACK), np.rad2deg(FALL_ANGLE_FWD)
        if y - gap - gauge_h >= floor:
            y -= gap + gauge_h
            self.gauge(mujoco.MjrRect(px, y, pw, gauge_h), f"Lean {tilt:+.0f} deg", -back, fwd, tilt,
                       [(-back, -0.7 * back, BAD), (-0.7 * back, -0.25 * back, WARN), (-0.25 * back, 0.25 * fwd, GOOD),
                        (0.25 * fwd, 0.7 * fwd, WARN), (0.7 * fwd, fwd, BAD)], f"back {back:.0f}", f"forward {fwd:.0f}", nls)
        # 4. body height gauge: how close the body is to collapsing
        height = p.info.get("body_height", 0.0) / p.sim.standing_height if p.steps else 1.0
        if y - gap - gauge_h >= floor:
            y -= gap + gauge_h
            self.gauge(mujoco.MjrRect(px, y, pw, gauge_h), f"Body height {height:.0%}", 0.6, 1.05, height,
                       [(0.6, COLLAPSE_FRACTION, BAD), (COLLAPSE_FRACTION, COLLAPSE_FRACTION + 0.12, WARN),
                        (COLLAPSE_FRACTION + 0.12, 1.05, GOOD)], f"collapse {COLLAPSE_FRACTION:.0%}", "standing", nls)
        # 5. scorecard
        score = self.scorecard(p)
        h = int(23 * nls) * len(score) + 2 * int(8 * nls)
        if y - gap - h >= floor:
            y -= gap + h
            draw_box(ctx, mujoco.MjrRect(px, y, pw, h), score, nls, False, top=True, widths=[pw - 2 * int(8 * nls) - 2], alpha=0.7)

    def scorecard(self, p):
        """How this episode is going (or went), as (text, level) lines."""
        def grade(value, good, warn):
            return GOOD if value <= good else WARN if value <= warn else BAD
        if not p.steps:
            return [("SCORECARD", "dimtitle"), ("Waiting for the first step", None), ("", None), ("", None), ("", None)]
        verdict = (f"{TAG[BAD]} FELL: the leg {p.cause}", BAD) if p.fell else \
                  (f"{TAG[GOOD]} SURVIVED", GOOD) if p.done else ("RUNNING", "info")
        lim = np.rad2deg(FALL_ANGLE_FWD) if p.max_tilt >= 0 else np.rad2deg(FALL_ANGLE_BACK)
        toe_level = grade(p.peak_force_toe(), 0.8 * LAMBDA_BW, LAMBDA_BW)
        lean_level = grade(abs(p.max_tilt) / lim, 0.25, 0.7)
        low = p.min_height
        height_level = BAD if low < COLLAPSE_FRACTION else WARN if low < COLLAPSE_FRACTION + 0.12 else GOOD
        return [("SCORECARD", "dimtitle"), verdict,
                (f"{TAG[toe_level]} Toe peak {p.peak_force_toe():.2f} of {LAMBDA_BW:.2f} BW", toe_level),
                (f"{TAG[lean_level]} Worst lean {p.max_tilt:+.0f} of {lim:.0f} deg", lean_level),
                (f"{TAG[height_level]} Body low {low:.0%} (min {COLLAPSE_FRACTION:.0%})", height_level)]

    def chart(self, rect, title, dt, now, lines, yrange, limit, p, yformat="%.0f"):
        """A small line chart with mjr_figure. lines: (values per step, colour). A white line marks the current time and a
        dim one each event (touchdown, peak load)."""
        fig = mujoco.MjvFigure()
        mujoco.mjv_defaultFigure(fig)
        fig.flg_legend = 0
        fig.title = title
        fig.xformat = "%.2f"                                                    # seconds
        fig.yformat = yformat
        fig.xlabel = "s"
        fig.range[:] = [[0.0, MAX_EPISODE_LENGTH * dt], [yrange[0], yrange[1]]]
        fig.gridsize[:] = [3, 3]                                                # ticks at 0, 0.25 and 0.5 s
        fig.figurergba[:] = [0.05, 0.06, 0.10, 0.75]
        fig.panergba[:] = [0.08, 0.10, 0.16, 0.55]
        fig.textrgb[:] = [0.85, 0.88, 0.92]
        fig.gridrgb[:] = [0.35, 0.38, 0.45]
        fig.linewidth = 2.0
        k = 0

        def add(xs, ys, rgb, width=None):
            nonlocal k
            n = min(len(xs), 1000)
            if k >= 90 or n < 2:
                return
            fig.linepnt[k] = n
            fig.linedata[k, 0:2 * n:2] = xs[:n]
            fig.linedata[k, 1:2 * n:2] = ys[:n]
            fig.linergb[k] = rgb
            k += 1
        total = MAX_EPISODE_LENGTH * dt
        if limit is not None:
            add([0.0, total], [limit, limit], (1.0, 0.55, 0.05))                    # the comfort limit, orange
        for step_i, rgb in ((p.touch_step, (0.6, 0.6, 0.7)), (p.peak_step if p.peak_fired else None, (0.6, 0.6, 0.7))):
            if step_i:
                add([step_i * dt] * 2, [yrange[0], yrange[1]], rgb)                  # event markers
        for values, rgb in lines:
            xs = [0.0] + [(j + 1) * dt for j in range(len(values))]
            add(xs, [0.0] + list(values), rgb)
        add([now, now], [yrange[0], yrange[1]], (1.0, 1.0, 1.0))                     # now
        mujoco.mjr_figure(rect, fig, self.note_ctx)

    def gauge(self, rect, title, lo, hi, value, zones, left_label, right_label, nls):
        """A horizontal gauge: coloured zones (green, yellow, orange), a white marker at the value, and the limits."""
        ctx, pad = self.note_ctx, int(8 * nls)
        mujoco.mjr_rectangle(rect, 0.05, 0.06, 0.10, 0.70)
        bx, bw = rect.left + pad, rect.width - 2 * pad
        by, bh_ = rect.bottom + int(rect.height * 0.28), int(rect.height * 0.30)
        def at(v):
            return bx + int(bw * (min(max(v, lo), hi) - lo) / (hi - lo))
        for a, b, level in zones:
            r, g, bl = LEVEL_COLOUR[level]
            mujoco.mjr_rectangle(mujoco.MjrRect(at(a), by, max(1, at(b) - at(a)), bh_), r, g, bl, 0.85)
        mujoco.mjr_rectangle(mujoco.MjrRect(at(value) - 2, by - int(3 * nls), 4, bh_ + int(6 * nls)), 1, 1, 1, 1)
        mujoco.mjr_rectangle(rect, 0, 0, 0, 0.0)                                      # back to the gauge's own viewport for the text
        for dx in (0, 1):
            mujoco.mjr_text(0, title, ctx, (pad + dx) / rect.width, 1 - (pad + 14 * nls) / rect.height, 1, 1, 1)
            mujoco.mjr_text(0, left_label, ctx, (pad + dx) / rect.width, 0.02, 0.8, 0.84, 0.9)
        right_w = len(right_label) * 11.2 * nls
        mujoco.mjr_text(0, right_label, ctx, max(0.0, (rect.width - pad - right_w) / rect.width), 0.02, 0.8, 0.84, 0.9)

    def word_width(self, word, FW, FH):
        """Width in pixels of a word (or phrase) in the note's font, measured once."""
        key = (self.nfs, word)
        if key not in self.text_widths:
            self.text_widths[key] = self.measure(word, FW, FH, self.note_ctx)
        return self.text_widths[key]

    def note_layout(self, paragraphs, inner_w, FW, FH):
        """Lay the note's paragraphs out in lines of `inner_w` pixels, justified: the gaps between the words of
        every line but the last of a paragraph are stretched so both edges are straight. Returns, per line, a list
        of (word, x) (None for a blank line). Worked out once and kept."""
        key = (self.nfs, inner_w, id(paragraphs))
        if key in self.note_cache:
            return self.note_cache[key]
        space = self.word_width("n n", FW, FH) - self.word_width("nn", FW, FH)        # width of a space
        out = []
        for para in paragraphs:
            out.append(None)
            lines, cur, cur_w = [], [], 0
            for word in para.split():
                ww = self.word_width(word, FW, FH)
                if cur and cur_w + space + ww > inner_w:
                    lines.append(cur)
                    cur, cur_w = [], 0
                cur_w += ww + (space if cur else 0)
                cur.append(word)
            lines.append(cur)
            for k, line in enumerate(lines):
                widths = [self.word_width(w, FW, FH) for w in line]
                gap = (inner_w - sum(widths)) / (len(line) - 1) if (k < len(lines) - 1 and len(line) > 1) else space
                x, placed = 0.0, []
                for w, ww in zip(line, widths):
                    placed.append((w, x))
                    x += ww + gap
                out.append(placed)
        self.note_cache[key] = out
        return out

    def draw_note(self, area, title, layout, inner_w, nls, alpha):
        """The motor-model note, justified, small and dim, in the top-left corner of `area`."""
        ctx = self.note_ctx
        lh, pad = int(23 * nls), int(8 * nls)
        w, h = inner_w + 2 * pad, lh * (1 + len(layout)) + 2 * pad
        mujoco.mjr_rectangle(mujoco.MjrRect(area.left, area.bottom + area.height - h, w, h), 0.05, 0.06, 0.10, alpha)
        def line_y(j):
            return 1 - (pad + lh * (j + 0.5) + 14 * nls) / h
        for dx in (0, 1):
            mujoco.mjr_text(0, title, ctx, (pad + dx) / w, line_y(0), *LEVEL_COLOUR["dimtitle"])
        for j, line in enumerate(layout, 1):
            for word, x in line or []:
                for dx in (0, 1):
                    mujoco.mjr_text(0, word, ctx, (pad + x + dx) / w, line_y(j), *LEVEL_COLOUR["dim"])

    def draw_keys(self, x0, y0, width, h, ls, status):
        """The key strip: each key on a bright yellow cap, its action next to it. mjr_label can colour text, and
        drawing each label twice, one pixel apart, makes the letters bold. Keys that do not fit are left out."""
        ctx = self.ctxs[0]
        cw = 14.6 * ls                                                  # width of one character of a label
        def label(x, w, text, bg, fg):
            r = mujoco.MjrRect(int(x), y0 + int(3 * ls), int(w), h - int(6 * ls))
            mujoco.mjr_label(r, 0, text, *bg, *fg, ctx)
            r2 = mujoco.MjrRect(int(x) + 1, r.bottom, int(w), r.height)
            mujoco.mjr_label(r2, 0, text, 0, 0, 0, 0, *fg, ctx)         # same text again, no background: bolder
        gap = int(8 * ls)
        status_w = (len(status) + 1.4) * cw
        x, end = x0 + gap, x0 + width - status_w - gap
        for key, text in KEY_CAPS:
            kw, tw = (len(key) + 1.6) * cw, (len(text) + 1.4) * cw
            if x + kw + tw > end:
                break
            label(x, kw, key, (1.0, 0.82, 0.0, 1.0), (0.0, 0.0, 0.0))
            label(x + kw, tw, text, (0.05, 0.08, 0.2, 0.9), (1.0, 1.0, 1.0))
            x += kw + tw + gap
        label(x0 + width - status_w - gap / 2, status_w, status, (0.0, 0.55, 0.85, 0.95), (1.0, 1.0, 1.0))

    def close(self):
        self.ctxs.clear()
        self.glfw.destroy_window(self.window)
        self.glfw.terminate()


def show_players(players, n, seconds, first_seed=1000, shot_dir="screenshots", fullscreen=True):
    """Replay n episodes (seeds first_seed, first_seed + 1, ...) in a 1080p MuJoCo window, in slow motion.
    With several players (--compare) they all play the same seed at the same time, so the ground is the same."""
    step_dt = players[0].sim.model.opt.timestep * FRAME_SKIP             # 5 ms of simulated time per env step
    ctl = Controls(seconds / (MAX_EPISODE_LENGTH * step_dt), help_on=False)   # slow-motion factor
    print(f"\nshowing {n} episodes in the MuJoCo viewer (a full episode takes {seconds:g} s); press ENTER in the window to start")
    view = View(players, ctl, fullscreen=fullscreen)
    cam_shown, idx, done_at = -1, 0, None
    started = [time.perf_counter()]                                      # when the episode on screen began

    def load(i):
        nonlocal idx, done_at
        idx, done_at = i % n, None
        ctl.paused, ctl.user_pause = True, False                    # nothing plays until ENTER
        # LEFT on the first goes to the last, RIGHT on the last to the first
        started[0] = time.perf_counter()
        for p in players:
            p.reset(first_seed + idx)

    def advance(announce=True):
        """One env step for every player that is still going."""
        for p in players:
            p.settled = p.settled or p.done                      # a finished player has shown its last state
        for p in players:
            if not p.done:
                p.step()
                if p.done and announce:
                    print(f"episode {idx + 1}/{n} (seed {p.seed}) {p.label}: "
                          f"{'fell, ' + p.cause if p.fell else 'survived'}, return {p.ret:.2f}, "
                          f"toe peak {p.toe:.2f} BW, played in {time.perf_counter() - started[0]:.1f} s")
        if ctl.autopause and any(p.new_event for p in players):          # stop at TOUCHDOWN, PEAK LOAD and FELL
            ctl.paused = ctl.user_pause = True
        for p in players:
            p.new_event = None

    try:
        load(0)
        last = time.perf_counter()
        progress = 0.0        # how far the picture is from the previous step (0) to the current one (1)
        while view.is_running():
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            if ctl.jump or ctl.restart:
                load(idx + ctl.jump)
                ctl.jump = 0
                ctl.restart = False
                progress = 0.0
            finished = all(p.done for p in players) and progress >= 1.0
            if ctl.space and not ctl.paused and not finished:           # SPACE while playing = pause
                ctl.paused = ctl.user_pause = True
            elif ctl.space or ctl.enter:                                # ENTER (or SPACE when a prompt is up)
                if finished:                                            # next episode, or back to the first after the last
                    load(idx + 1)
                    progress = 0.0
                if ctl.paused or finished:
                    if max(p.steps for p in players) == 0:
                        started[0] = now                                # the clock for "played in" starts at play
                    ctl.paused = ctl.user_pause = False
            ctl.space = ctl.enter = False
            if ctl.step_req:                                             # frame by frame: , and . pause and step
                ctl.paused = ctl.user_pause = True
                req, ctl.step_req = ctl.step_req, 0
                if req > 0:
                    if not all(p.done for p in players):
                        advance()
                else:                                                    # no undo in the physics: replay the same seed to one step earlier
                    target = max(0, max(p.steps for p in players) - 1)
                    for p in players:
                        p.reset(first_seed + idx)
                        while p.steps < target and not p.done:
                            p.step()
                progress = 1.0 if max(p.steps for p in players) else 0.0
            if not ctl.paused:
                progress += dt / (step_dt * ctl.slow)                    # real time passed -> steps of simulated time
                while progress >= 1.0 and not all(p.done for p in players):
                    advance()
                    progress -= 1.0
                    if ctl.paused:                                           # auto-pause fired
                        break
                progress = min(progress, 1.0)
            for p in players:                                            # draw between the last two steps (smooth)
                p.pose(1.0 if p.settled else progress)
            all_done = all(p.done for p in players) and progress >= 1.0
            if all_done:                                                 # what ENTER will do now
                prompt = "Press ENTER to reset" if idx + 1 >= n else "Press ENTER to continue"
            elif ctl.paused:
                at_start = max(p.steps for p in players) == 0
                prompt = "Press ENTER to play" if at_start or not ctl.user_pause else "Press SPACE to continue"
            else:
                prompt = None
            if ctl.help:
                prompt = None                                            # the key list is open: it takes the middle of the window
            if all_done:
                ctl.paused = True
            if cam_shown != ctl.camera % len(CAMERAS):
                view.set_camera(ctl.camera)
                cam_shown = ctl.camera % len(CAMERAS)
            ep = f"{idx + 1}/{n}  (seed {first_seed + idx})"
            save_to = None
            if ctl.screenshot:
                ctl.screenshot = False
                save_to = Path(shot_dir) / f"stance_{players[0].label.replace(' ', '_')}_seed{first_seed + idx}_step{players[0].steps}.png"
            view.render([p.rows(ep) for p in players],
                        (lambda i, w: inference_lines(players[i], w)),
                        HELP,
                        f"{1 / ctl.slow:.2f}x  {CAMERAS[ctl.camera % len(CAMERAS)][0]}", save_to, prompt)
            if save_to:
                print(f"screenshot saved: {save_to}")
            time.sleep(max(0.0, 0.008 - (time.perf_counter() - now)))   # no need to draw faster than ~120 fps
    finally:
        view.close()
        for p in players:
            p.env.close()
# --- viewer end ---


def record_player(player, seed, path, hud=True, seconds=10.0, fps=60):
    """Render one episode offscreen and save it as an mp4, with the HUD and force arrows drawn in. The motion is
    interpolated between the env steps, so a full episode lasts `seconds` and is smooth at `fps` frames/s."""
    import imageio
    sub = max(1, round(seconds * fps / MAX_EPISODE_LENGTH))             # frames drawn per env step
    renderer = mujoco.Renderer(player.sim.model, height=720, width=1280)
    cam = mujoco.MjvCamera()
    set_camera(cam, player.sim, 0)                                       # the same side view as the viewer
    player.reset(seed)
    with imageio.get_writer(path, fps=fps) as video:                     # written frame by frame, not kept in memory
        def frame():
            video.append_data(render_frame(renderer, player, cam, f"seed {seed}", hud))
        frame()
        while not player.done:
            player.step()
            for k in range(1, sub + 1):
                player.pose(k / sub)
                frame()
        for _ in range(fps):                                             # hold the last frame for a second
            frame()
    renderer.close()


# ---------------------------------------------------------------------------------------------------------------------
# Hooking into demo.py
# ---------------------------------------------------------------------------------------------------------------------
UI_FLAGS = """viewer and video flags (handled by demo_ui.py; demo.py's own flags, listed below, work as before):
  --compare            viewer: the best fixed action and every chosen network side by side, on the same ground
  --ground train|test  viewer and video: which ground to show (default train; the test ground sinks much deeper)
  --seed S             viewer and video: first episode seed to show (default 1000; the evaluation always uses 1000-1199)
  --seconds S          viewer and video: how long one full episode takes to play (default 10)
  --windowed           viewer: a 1920x1080 window instead of full screen
  --screenshots DIR    where the P key saves screenshots (default screenshots/)
  --no-hud             video: leave the overlays out
  --classic-viewer     use demo.py's original viewer and video instead"""


def _short(net):
    """'networkB_lstm' -> 'lstm'."""
    return net.NAME.split("_")[-1]


def install(demo):
    """Called from the block at the end of demo.py, before main() runs. `demo` is demo.py's own module."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--compare", action="store_true")
    p.add_argument("--ground", choices=["train", "test"], default="train")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--windowed", action="store_true")
    p.add_argument("--screenshots", default="screenshots")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--classic-viewer", action="store_true")
    ui, rest = p.parse_known_args(sys.argv[1:])
    sys.argv[1:] = rest                                    # demo.py's own parser only sees its own flags
    if "-h" in rest or "--help" in rest:
        print(UI_FLAGS + "\n")
    # Remember every run main() loads or trains, so the compare view can show them all (and the progress knows how many).
    runs = []
    load_run, train_run = demo.load_run, demo.train_run

    def load_run_kept(run_dir):
        r = load_run(run_dir)
        cfg = json.loads((Path(run_dir) / "config.json").read_text())["config"]
        r[2].frames = cfg.get("frames", 1)                    # how many readings the network sees at once (for the banner)
        runs.append((_short(r[0]), Path(run_dir).name[len("final_s"):], r))
        return r

    def train_run_kept(network, steps=None, seed=0):
        r = train_run(network, steps, seed)
        r[2].frames = {**demo.TRAIN, **r[0].CONFIG}.get("frames", 1)
        runs.append((_short(r[0]), "new", r))
        return r

    def show(net, model, vn, wrappers, n, slow, first_seed=1000):
        """Takes the place of demo.show_episodes (main() passes the first run; slow is replaced by --seconds)."""
        try:
            seed = ui.seed if ui.seed is not None else first_seed
            if ui.compare:                                     # the best fixed action, then the first run of each network
                players, seen = [Player("reference", (), fixed=True, ground=ui.ground)], set()
                for name, ms, (n_, m_, v_, w_) in runs:
                    if name not in seen:
                        seen.add(name)
                        players.append(Player(f"{name} s{ms}", w_, n_, m_, v_, ground=ui.ground))
            else:
                ms = next((ms for _, ms, r in runs if r[1] is model), "")
                players = [Player(f"{_short(net)} s{ms}", wrappers, net, model, vn, ground=ui.ground)]
            show_players(players, n, ui.seconds, seed, ui.screenshots, not ui.windowed)
        except Exception:
            traceback.print_exc()
            raise

    def record(net, model, vn, wrappers, seed, path):
        """Takes the place of demo.record_episode: 60 fps, interpolated, with the overlays."""
        seed = ui.seed if ui.seed is not None else seed
        ms = next((ms for _, ms, r in runs if r[1] is model), "")
        record_player(Player(f"{_short(net)} s{ms}", wrappers, net, model, vn, ground=ui.ground), seed, path,
                      not ui.no_hud, ui.seconds)
        print(f"video: {ui.ground} ground, seed {seed}, {ui.seconds:g} s per episode")

    demo.load_run, demo.train_run = load_run_kept, train_run_kept
    demo.run_episodes = _with_progress(demo.run_episodes, lambda: 2 * (1 + len(runs)))   # also with --classic-viewer
    if ui.classic_viewer:
        return
    ground_module.GroundModel.update_visual = _update_visual   # drawing only
    demo.show_episodes, demo.record_episode = show, record


def _with_progress(run_episodes, calls):
    """Wrap common.evaluate.run_episodes so the evaluation in main() shows how far it has got. It plays the same episodes
    in the same order, one seed at a time (each episode resets the env and the policy, so the results are identical).
    calls() is how many times main() will call it: once per ground for the reference and for every run."""
    state = {"done": 0, "t0": None, "shown": -1}
    tty = sys.stdout.isatty()

    def counted(policy, env, seeds, *args, **kwargs):
        if state["t0"] is None:
            state["t0"] = time.time()
        total = max(calls() * len(seeds), state["done"] + len(seeds))
        out = []
        for s in seeds:
            out += run_episodes(policy, env, [s], *args, **kwargs)
            state["done"] += 1
            pct = min(100, int(100 * state["done"] / total))
            elapsed = time.time() - state["t0"]
            eta = elapsed / state["done"] * (total - state["done"])
            line = f"evaluating: {pct:3d}% ({state['done']} of {total} episodes, about {int(eta // 60)} min {int(eta % 60):02d} s left)"
            if tty:                                            # one line, updated in place
                print("\r" + line + " " * 4, end="" if state["done"] < total else "\n", flush=True)
            elif pct // 10 != state["shown"] // 10 or state["done"] == total:   # a log file: a line every 10%
                print(line, flush=True)
            state["shown"] = pct
        return out
    return counted
