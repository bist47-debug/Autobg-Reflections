# AutoBG Reflections

Automatic, perspective-matched floor reflections for car images.

Upload a car and a backdrop, and the script places a reflection on the floor that
follows the car's angle and meets each tyre where it touches the ground. It runs as a
Gradio app inside Google Colab.

## Files

| File | What it is |
|---|---|
| `car_reflection_colab.py` | The working version. Wheel detection uses Ultralytics YOLOv8 (Open Images weights). |
| `car_reflection_colab_owlv2.py` | Same script with the wheel detector swapped for OWLv2. Not yet verified on real cars. |

## How to run

1. Open a new notebook in [Google Colab](https://colab.research.google.com).
2. Paste the whole of `car_reflection_colab.py` into one cell and run it.
3. Wait for the line `Running on public URL: https://....gradio.live` and open that link.

The first run takes a few minutes: it installs the packages and downloads the cutout
and wheel-detection models. A GPU runtime makes each image faster but is not required.

If the cell fails with a NumPy import error, choose **Runtime → Restart session** and
run the cell again.

## Using the app

1. **Car image**: a transparent PNG, or any photo (the car is cut out automatically).
2. **Backdrop**: the scene to place the car on. It is resized to the car image's size.
3. The reflection is generated as soon as both are loaded.

The preview shows what was detected: red dots are the tyre contacts, and the green
line is the floor line the car is mirrored about.

### Controls

| Control | What it does |
|---|---|
| Car image already has a shadow | Ignores a shadow baked into the image when finding the tyres, then draws it back over the reflection. Leave on for pipeline output that already has shadows; turn off for clean cutouts. |
| What to reflect | **Whole car**, or **Wheels only** (tyres and rims, no bumper or body). |
| Reflection opacity | Overall strength of the reflection. |
| Fade length | How far the reflection extends before fading out, as a fraction of the car's height. |
| Blur | Softness. The reflection is sharp at the tyres and blurrier further away. |
| Blend mode | Normal, Screen, Soft light, Overlay, Hard light, Lighten, Multiply (same maths as Photoshop layer modes). |
| Shine | Adds a glow from the bright parts of the reflection, for a glossy-floor look. |

## How it works

1. **Cut out the car** with rembg, unless the upload is already transparent.
2. **Remove any existing shadow** from the measurements by cutting the car out a second time.
3. **Detect the wheels**, including a tyre that only shows under a bumper.
4. **Find each tyre's contact line** from the tyre's own bottom edge.
5. **Work out the floor under the body**: along the side from the two same-side tyres,
   and under the front or rear from the bumper's bottom edge. Where no tyre shows under
   a bumper, the floor is placed slightly below it so the reflection does not touch the bumper.
6. **Join these into one continuous floor line** and mirror the car about it, column by column.
   Under each wheel the line is that tyre's own contact line, so the wheel reflects directly beneath itself.
7. **Finish**: fade with distance, blur progressively, let floor detail show through,
   then composite floor, reflection, shadow and car in that order.

## Input requirements

- **Leave room below the car.** The car stays where it is in the image, so a car close to
  the bottom edge has no space for its reflection.
- **Use a clean cutout.** Stray objects left attached to the car (railings, other vehicles)
  throw off the floor line.

## Known limitations

- The gap between a bumper and the floor is an estimate, so on very low or very high
  bumpers the reflection can start slightly too close or too far.
- Near each wheel the reflected bodywork bends slightly, where the floor line eases
  between the tyre and the body.
- In Wheels only mode, a wheel mostly hidden by the body can bring a strip of bodywork with it.
- Tested on roughly 35 cars across front, rear, side and three-quarter views.

## Licensing

Check these before any commercial use.

- **Ultralytics YOLO** (used in `car_reflection_colab.py`) is released under AGPL-3.0.
  Ultralytics states that commercial, SaaS and closed-source use requires their
  Enterprise licence. See <https://www.ultralytics.com/license>.
- **OWLv2** (used in `car_reflection_colab_owlv2.py`) and the transformers library are
  understood to be Apache-2.0. Confirm on the model's page.
- **rembg**, OpenCV, NumPy and Gradio use permissive licences. The cutout model that
  rembg downloads has its own licence, which should be confirmed separately.
