# Technical steps

How `car_reflection_colab.py` builds a reflection, in execution order, with the
function that does each step.

## 1. Cutout: `to_rgba`

- If the upload's alpha channel has values below 250, it is used directly.
- Otherwise rembg runs with the `isnet-general-use` model and returns RGBA.

## 2. Shadow removal: `strip_shadow`

- The RGBA image is composited over white and passed through rembg a second time.
- The car-only alpha is `min(original alpha, second-pass alpha)`.
- This copy is used for all measuring and mirroring. The original is kept for final drawing.

## 3. Wheel detection: `detect_wheels`

- The car is composited over flat grey (190) and cropped to its bounding box plus a 6% margin.
- YOLOv8m with Open Images V7 weights runs at `imgsz=1024`, `conf=0.08`, class-agnostic NMS,
  restricted to the `Wheel` and `Tire` classes.

## 4. Geometry analysis: `analyse`

- **Mask and envelope:** `alpha > 127` gives the mask. The lowest mask pixel per column gives
  the bottom envelope.
- **Box clean-up:** boxes overlapping by more than 50% of the smaller one are merged. A box is
  kept only if it is in the lower half of the car, at least 4% of car height, and its bottom
  lies within 25% of its height of the envelope.
- **Classification:** a box with width/height below 1.4 and at least half the tallest wheel's
  height is a full wheel. Anything else is a tyre peeking under a bumper. Two full wheels at
  both extremes of the car means a straight front or rear view.
- **Side line:** the line tangent to both same-side tyres from below, refined over three iterations.
- **Bumper edge:** a RANSAC line fit (400 iterations, tolerance 1.2% of car height) to the
  envelope beyond the corner wheel, with wheel columns excluded.
- **Bumper floor line:** same slope as the bumper edge, shifted down to the lowest thing under
  the bumper. If no tyre shows there, it is shifted a further `END_GAP` = 12% of wheel height.
- **Missed tyres:** connected components hanging below the body edge that no box covers are
  added as extra wheels.
- **Per-tyre contact line (`tyre_edge`):** the longest underside edge of the convex hull of the
  tyre's bottom outline, with slope limited to ±0.45. A round bottom falls back to a level line
  at the lowest point.

## 5. Single floor line: `body_floor`, `floor_line`

- Body floor per column: `B(x) = min(side line, bumper line)`.
- Under each wheel, `B(x)` is replaced by that tyre's contact line, raised by 1.5% of wheel
  height to close any gap.
- The switch is weighted by a smoothstep over a margin of 30% of the wheel width, which keeps
  `B(x)` continuous.

## 6. Mirroring: `make_reflection`

- `cv2.remap` with `map_x = x`, `map_y = 2·B(x) − y`, bilinear, transparent border.
- In wheels-only mode (`wheels_only`), the source alpha is first multiplied by a feathered
  rounded-rectangle mask per wheel box.

## 7. Fade and blur

- Distance below the line: `d = (y − B(x)) / (fade × car height)`, clipped to 0..1.
- Alpha is multiplied by `(1 − d)^1.5`.
- Three Gaussian blurs of the premultiplied image (sigma `0.25·blur`, `blur`, `2.5·blur`) are
  interpolated by `d`, so blur increases with distance.

## 8. Compositing: `floor_texture`, `blend_pixels`, `composite`

- **Floor texture:** backdrop luminance divided by its Gaussian blur (sigma 12), clipped to
  0.55..1, multiplies the reflection so floor detail shows through.
- **Blend:** the reflection colour is un-premultiplied, blended with the backdrop by the chosen
  mode's formula, and mixed in by the reflection alpha.
- **Shine:** a Screen blend of the reflection's highlights, weighted by
  `clip((max channel − 0.45) / 0.55)^1.5`.
- **Top layer:** the original image (car plus shadow) is alpha-composited on top.

## 9. Preview: `draw_debug`

- Draws `B(x)` in green and each tyre contact as a red dot.

## Tunable constants

| Constant | Value | Meaning |
|---|---|---|
| `END_GAP` | 0.12 | Gap below a bumper with no tyre under it, as a fraction of wheel height |
| `FACE_ON_ASPECT` | 0.70 | Wheel width/height above which a wheel counts as mostly face-on |
| `SIDE_ON_ASPECT` | 0.90 | Mean wheel width/height above which the car counts as a side view |
| `DEFAULT_CLEARANCE` | 0.28 | Body-to-floor gap as a fraction of wheel height when it cannot be measured |
| `END_ON_CLEARANCE` | 0.12 | Body-to-floor gap as a fraction of car width in straight front/rear views |
