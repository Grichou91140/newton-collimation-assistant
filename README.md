# Newton Collimation Assistant

A Python tool for **assisted collimation of Newtonian telescopes** using a camera placed in the focuser.

The script estimates the **geometric center of the focuser tube**, averages measurements over several camera rotations, and provides a **live review mode** to evaluate the final collimation quality.

This tool was designed for **practical field use**, especially during night sessions when traditional tools such as a Cheshire eyepiece may be difficult to use.

---

## Features

- Interactive **3-point circle initialization**
- User-defined **analysis sector** to exclude problematic areas
- Edge detection of the **focuser tube**
- **Multi-angle acquisition** (typically 0°, 90°, 180°, 270°)
- Averaged estimation of the focuser center
- **Zoomed review mode (x2)**
- Two adjustable reference circles
- **5-second live analysis** of the optical center
- Estimated metrics:
  - measured collimation error
  - stability of the error
  - collimation confidence

---

## Hardware

This software was tested using a camera collimation tool based on the **DCAL Camera Collimator for Reflecting Telescopes** design by **Dave Aldrich**.

Project page:

https://www.printables.com/model/1232771-dcal-camera-collimator-for-reflecting-telescopes

The hardware design is not part of this repository.

This project only provides the **software analysis tool** used with the camera.

---

## Why this tool?

Cheap laser collimators are convenient but often suffer from:

- poor internal collimation
- mechanical play in the focuser
- inconsistent seating
- lack of quantitative feedback

This tool instead:

- estimates the reference center directly from the **camera image**
- works with the **actual optical geometry**
- provides **visual and numerical feedback**

In practice it works well as a **field collimation assistant**, especially when combined with a final star test or Cheshire verification.

---
First Step :
![Main](examples/1-MAIN-DETECTION-WINDOW.png)
Detection : 
shortcut m (multiple angles), 
then p (clic 3 points), 
then u (again 3 points to define sector to analyse)
then d (detect focuser tube)
then n (new detection)
turn the camera 90°
![Detection window](examples/2-DETECTION.png)
After 4 angles, 0°, 90°, 180°, 270° 
![After 4 detections](examples/3-LAST-DETECTION.png)
shortcut b brings to review window
![Setup window](examples/4-REVIEW-WINDOW.png)
Make your collimation based on the detected center wich is the geometrical center of the focuser tube
![Done](examples/5-COLLIMATION.png)
Then t and turn VERY GENTLY the camera, after 5s you'll get the result.
![Test and Confidence](examples/6-FINAL_TEST_and_RESULTS.png)

---

## Keywords

Newton telescope  
Newtonian collimation  
Telescope collimation  
Astrophotography tools  
Focuser alignment  
Camera collimation  
Reflector telescope collimation

---

## Requirements

- Python 3.9+
- OpenCV
- NumPy

Install dependencies:

```bash
pip install -r requirements.txt
