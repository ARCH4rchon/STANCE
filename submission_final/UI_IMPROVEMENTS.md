# UI improvements to the STANCE demo

The improved viewer and video from the `UI_Improvements` branch, added to `Final_Submission` **without changing any
existing code**. `demo.py` only gains one clearly marked block at its end, which hands the viewer and the video over to
the new file `demo_ui.py`. **The physics, the reward, the trained networks and the evaluation numbers are unchanged:**
the per-episode CSV is byte-for-byte the same with and without the new viewer, and the leg moves identically in the
visual-only model (checked on three seeds, maximum difference 0).

Run it from `submission_final/Network_Agent` (conda env `stance`):

```bash
python demo.py --compare --ground test --seed 1108 --episodes 5 --show 1 --model-seed 0       # three controllers, side by side
python demo.py --network lstm --model-seed 0 --ground test --seed 1120 --episodes 5 --show 3   # one network, a hard ground
python demo.py --record demo.mp4 --show 0 --seconds 10                                         # a 60 fps video
python demo.py --classic-viewer                                                                # the original viewer and video
```

`python demo.py --help` lists the viewer's flags first, then `demo.py`'s own.

## How it is wired in

| File | What changed |
|---|---|
| `Network_Agent/demo.py` | **11 lines added at the end, none changed or removed.** When run as a script, it calls `demo_ui.install()` before `main()` |
| `Network_Agent/demo_ui.py` | New file: the whole viewer, the overlays and the video |
| `UI_IMPROVEMENTS.md` | This file |

`demo_ui.install()`:

- takes the viewer's own flags off the command line, so `demo.py`'s parser sees only its own flags:
  `--compare`, `--ground train|test`, `--seed S`, `--seconds S`, `--windowed`, `--screenshots DIR`, `--no-hud`,
  `--classic-viewer`. `--seed` only picks which episode the viewer and the video show. The evaluation always uses
  seeds 1000-1199, as before. The line `main()` prints after recording still says seed 1000, so the video's real
  ground and seed are printed on the line before it;
- replaces `demo.show_episodes` and `demo.record_episode`, which `main()` calls, and keeps track of the runs
  `main()` loads so the compare view can show them all;
- draws the scene with a visual-only copy of `Gym/model/leg.xml`, written at run time to a temporary folder, plus a
  smoother drawing of the dented ground (`GroundModel.update_visual`, drawing only). The Gym files themselves are not
  changed. The copy only adds a sky, a floor texture, shadows, a light and finer non-colliding ground cells.
- does nothing with `--classic-viewer`, so the original `show_episodes` and `record_episode` run as before.

## Window and playback

- **Own 1920×1080 window**, full screen by default (`--windowed` for a normal window, `F11` to switch). On a taller screen
  the picture is letterboxed to 16:9. The window warms up before the first frame so it never opens clipped. Mouse drag
  orbits the camera and the wheel zooms.
- **Nothing plays by itself.** A pulsing pink banner says what ENTER will do: *Press ENTER to play*, *continue* (next
  episode), or *reset* (back to episode 1 after the last). After a pause with SPACE it says *Press SPACE to continue*.
  A second line reminds you that `,` and `.` step the simulation.
- **Playback speed:** `--seconds S` (default 10) sets how long a full 100-step episode takes to play, about 0.05× real
  time. It replaces `--slow`. `UP`/`DOWN` change the speed while running, down to 0.005×.
- **Smooth motion:** the picture is interpolated between the 5 ms simulation steps, about 70 frames a second instead of
  one per step. The physics itself never sees the blended states.
- **Frame-by-frame:** `,` steps back, `.` steps forward (both pause). There is no undo in the physics, so stepping back
  replays the same seed to one step earlier.
- **Episodes:** `LEFT`/`RIGHT` go to the previous or next episode (wrapping around inside `--show N`), `R` restarts.
- **Four cameras** (`V`): side, 3/4 view, front, and a foot close-up that follows the foot down as the ground gives way.
  The other three centre between the foot and the body so a tipping leg stays in the picture.
- **Key strip:** yellow key caps along the bottom are always visible, with the speed and camera on the right.
  `F1` opens the full key list in the middle of the window. `H` hides all panels, `I` the ankle's motor box and the
  note, `L` the legend, `T` the data panel, `P` saves a screenshot (`--screenshots DIR`), `G` hides the ghost leg, `A` turns on the auto-pause at events, `ESC` quits.

## Compare mode (`--compare`)

- One window with a column per controller (the best fixed action, then each chosen network), all on the same seed and ground.
- **Ghost reference leg** (`G`): the reference's leg is drawn as a translucent shell over each network's leg, so
  differences in posture show directly.
- **Network banner** along the top of every column says what kind of controller it is: the reference is a fixed rule,
  the MLP looks back over the last 10 moments with no memory, the LSTM sees one moment but remembers what came before.
- `--ground train|test` chooses the ground. The test ground sinks much deeper.

## HUD and explanation overlays

- **Stats panel:** controller, episode and seed, step, **sim time** (`0.200 / 0.500 s`), return, toe force, peak toe,
  ankle target, stiffness K, damping B, ground stiffness, dent depth, status (and *FELL: tipped over / collapsed*).
- **Legend** (`L`) explains every colour, arrow, ring and needle. It starts hidden when the window or columns are narrow.
- **"What the ankle's motor is doing" box** (`I`): five plain-words rows (*Ground*, *Ankle feels*, *Ankle's motor does*,
  *Setting*, *Decision*), written from the ankle's point of view, with no numbers. Each row is colour-coded:
  green = acceptable, yellow = going the right way but not ideal, orange = needs to change, **and** starts with a
  symbol (`[OK]`, `[~]`, `[!]`) so it does not rely on colour alone. The box keeps one fixed size however the text
  changes. It is a rule-based reading of tilt, forces and the chosen action, not the network's own reasoning.
- **Motor-model note** (small, dim, justified, under the stats panel): the motor is a virtual spring and damper,
  `torque = stiffness × (target angle − ankle angle) − damping × ankle speed`, limited to 150 N·m. The target angle is
  learned, not calculated: every 5 ms the network reads 11 sensor values and nudges a fixed starting setting, moving the
  target by at most 6°. Checked against the code and the saved training configs.
- **Chips under the banner:** *STIFFENING* (magenta) and *SOFTENING* (blue) when the commanded stiffness moves by more
  than 3% in a step, and event chips *TOUCHDOWN*, *PEAK LOAD x.x BW* and *FELL*. `A` pauses automatically at each event.
- **Data panel** (`T`, off by default): ground-force and motor-torque charts against time (a white "now" line, the
  1.58 BW comfort limit, event markers), a lean gauge (−15° to +35° fall limits), a body-height gauge (collapse at 70%),
  and a live **scorecard** (survived or fell, toe peak vs 1.58 BW, worst lean, lowest body height). Items that do not
  fit a small window are dropped.

## The scene

- **Look:** sky gradient, checker floor, soft shadows and a fill light (in the visual-only copy of `leg.xml`).
- **Ground:** coloured by stiffness (sand = soft, slate = hard); 280 cells of 5 mm instead of 70 of 2 cm, with
  rounded trench edges, a low rim of displaced soil, and darker colour where it is pressed. The strip is 14 cm wide.
- **Leg:** turns red when it falls.
- **Force arrows:** toe (red) and heel (violet) ground forces, thin, fully saturated, with a head that keeps its shape
  at any length. Length is 0.35 m per body weight (exaggerated) and a translucent **ghost arrow** shows the longest an
  arrow can be (2.3 BW).
- **Ankle motor drawn on the leg:** a teal cylinder at the ankle joint; a ring around it that grows with stiffness
  (grey, magenta while stiffening up, blue while softening); a cyan arc inside it showing the motor torque vector (which
  way it turns, growing with torque); and two needles, **white = target angle**, **dark = actual ankle angle**.
  Drawing only: no physics change. This is a virtual spring and damper, not a brake.

## Video recording (`--record`)

- 1280×720 at **60 fps**, interpolated, lasting `--seconds` (default 10) plus a one-second hold; frames are written to
  disk as they render (no memory build-up). `--no-hud` leaves the overlays out.
- Videos include the stats panel, banner, legend, motor note, inference box, scorecard, chips, motor, needles and
  arrows. The charts, gauges and ghost leg are viewer-only.

## Known limits

- Windows below about 1280×720 are crowded. At 800×450 only part of the data panel fits. Full screen at 1080p is the
  tested layout.
- Everything was tested with scripted key presses and saved frames, not by watching the live window animate.
- The *PEAK LOAD* event fires once, when the ground force first falls below 90% of its peak. The grading thresholds in
  the inference box and the scorecard (for example 25% and 70% of a tilt limit) are the authors' choices.
- Startup prints harmless `libdecor-gtk` and `Gtk` warnings.
