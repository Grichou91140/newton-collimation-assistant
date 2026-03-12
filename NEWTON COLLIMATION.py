__version__ = "1.0.0"
"""
Newton Collimation Assistant
============================

Overview
--------
This script helps estimate the focuser-tube center from a camera view, then
evaluate collimation using a short live analysis.

Main workflow
-------------
1. Start multi-angle acquisition with:
      m
2. For each camera rotation angle (0°, 90°, 180°, 270°):
      p   -> click 3 points on the focuser-tube edge to define the initial circle
      u   -> click 3 points to define the analysis sector
      d   -> run single-frame edge detection
      n   -> save the current measurement
3. After the 4 measurements:
      b   -> open review mode
4. In review mode:
      - adjust the 2 circle diameters with the trackbars
      t   -> run a 5-second live analysis while gently rotating the camera
      o   -> return to normal view

What the script shows
---------------------
Normal view:
- Initial circle from 3 clicked points
- Analysis sector
- Detected edge points
- Detected circle / ellipse
- Mean center from all saved multi-angle measurements

Review mode:
- x2 zoom centered on the final mean center
- Full crosshair through the center
- Two adjustable concentric circles to help collimating
- Adjust your primary and secondary to align them on the mean center
- Measured collimation error
- Error stability over 5 seconds
- Collimation confidence score

Keyboard shortcuts
------------------
General:
    ESC   Quit
    a/z   Decrease / increase brightness
    s/x   Decrease / increase contrast
    g/h   Decrease / increase gradient threshold
    j/k   Decrease / increase radius tolerance
    f     Freeze current frame
    r     Reset everything
    c     Cancel current pointing mode

Measurement:
    p     Start 3-point initial circle selection
    u     Start 3-point analysis sector selection
    d     Run single-frame detection
    n     Save current measurement
    m     Start multi-angle acquisition
    b     Open review mode

Review mode:
    t     Run 5-second live collimation analysis
    o     Return to normal view

Practical tips
--------------
- Use stable lighting. I use a phone light placed 90° of the secondary support in the tube.
- Keep the analysis sector tight around the highest-contrast part of the tube edge.
- During the 5-second review analysis, very gently rotate the camera to assess stability.
- A low measured collimation error with low stability error generally indicates
  a very good result.
- Confirm your collimation with a Cheshire by daylight.
- On the field, by night, confirm with star test.

Requirements
------------
- Python 3
- OpenCV
- NumPy

Install:
    pip install opencv-python numpy
"""

import cv2
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)

brightness = 0
contrast = 1.0

circle_points = []
sector_points = []

pointing_circle_mode = False
pointing_sector_mode = False
mouse_pos = None

detected_center = None
detected_radius = None
detected_edges = []
filtered_edges = []
detected_ellipse = None
last_frozen_frame = None

# Focuser tube edge detection parameters
N_ANGLES = 180
SEARCH_HALF_WIDTH = 60
GRADIENT_THRESHOLD = 8.0
RADIUS_TOLERANCE = 12.0

# Multi-angle acquisition
multi_mode = False
multi_angles = [0, 90, 180, 270]
multi_index = 0
multi_results = []

# Review mode
show_review_view = False
REVIEW_WINDOW = "Collimation Review"
circle1_diameter = 120
circle2_diameter = 240
trackbars_created = False

# Non-blocking 5 s analysis
ANALYSIS_DURATION_SEC = 5.0
analysis_running = False
analysis_start_time = None
analysis_optical_centers = []
review_analysis = None


def noop(x):
    pass


def ensure_review_trackbars():
    global trackbars_created
    if not trackbars_created:
        cv2.namedWindow(REVIEW_WINDOW)
        cv2.createTrackbar("Circle 1 diameter", REVIEW_WINDOW, circle1_diameter, 1200, noop)
        cv2.createTrackbar("Circle 2 diameter", REVIEW_WINDOW, circle2_diameter, 1200, noop)
        trackbars_created = True


def read_review_trackbars():
    global circle1_diameter, circle2_diameter
    if trackbars_created:
        circle1_diameter = cv2.getTrackbarPos("Circle 1 diameter", REVIEW_WINDOW)
        circle2_diameter = cv2.getTrackbarPos("Circle 2 diameter", REVIEW_WINDOW)
        circle1_diameter = max(2, circle1_diameter)
        circle2_diameter = max(2, circle2_diameter)


def circle_from_3pts(p1, p2, p3):
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    p3 = np.array(p3, dtype=float)

    v1 = p2 - p1
    v2 = p3 - p1

    d = 2 * (v1[0] * v2[1] - v1[1] * v2[0])
    if abs(d) < 1e-6:
        return None, None

    ux = ((np.linalg.norm(v1) ** 2 * v2[1] -
           np.linalg.norm(v2) ** 2 * v1[1]) / d)
    uy = ((np.linalg.norm(v2) ** 2 * v1[0] -
           np.linalg.norm(v1) ** 2 * v2[0]) / d)

    center = p1 + np.array([ux, uy])
    radius = np.linalg.norm(center - p1)

    return center.astype(np.float32), float(radius)


def fit_circle_least_squares(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 3:
        return None, None

    x = pts[:, 0]
    y = pts[:, 1]

    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)

    try:
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None, None

    A_, B_, C_ = sol
    cx = -A_ / 2.0
    cy = -B_ / 2.0
    r2 = cx**2 + cy**2 - C_

    if r2 <= 0:
        return None, None

    return np.array([cx, cy], dtype=np.float32), float(np.sqrt(r2))


def fit_ellipse_if_possible(pts):
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) < 5:
        return None

    pts_cv = pts.reshape((-1, 1, 2)).astype(np.float32)
    try:
        return cv2.fitEllipse(pts_cv)
    except cv2.error:
        return None


def preprocess_gray(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=contrast, beta=brightness)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def normalize_angle(a):
    return a % (2 * np.pi)


def point_angle(center, pt):
    return normalize_angle(math.atan2(pt[1] - center[1], pt[0] - center[0]))


def angle_is_between_ccw(a, start, end):
    if start <= end:
        return start <= a <= end
    return a >= start or a <= end


def build_sector_from_3pts(center, pts):
    if center is None or len(pts) != 3:
        return None

    a1 = point_angle(center, pts[0])
    a2 = point_angle(center, pts[1])
    a3 = point_angle(center, pts[2])

    if angle_is_between_ccw(a3, a1, a2):
        return a1, a2
    return a2, a1


def angle_in_sector(theta, sector):
    if sector is None:
        return True
    start, end = sector
    return angle_is_between_ccw(normalize_angle(theta), start, end)


def draw_cross(img, center, color, size=10, thickness=2):
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def draw_full_cross(img, center, color=(255, 255, 255), thickness=1):
    h, w = img.shape[:2]
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (0, y), (w - 1, y), color, thickness)
    cv2.line(img, (x, 0), (x, h - 1), color, thickness)


def draw_sector_arc(img, center, radius, sector, color=(255, 0, 255), thickness=2):
    if sector is None:
        return

    cx, cy = int(center[0]), int(center[1])
    start, end = sector

    start_deg = np.degrees(start)
    end_deg = np.degrees(end)

    if end_deg < start_deg:
        end_deg += 360.0

    cv2.ellipse(img, (cx, cy), (int(radius), int(radius)), 0.0, start_deg, end_deg, color, thickness)


def detect_circle_from_seed(frame, seed_center, seed_radius, sector=None,
                            n_angles=N_ANGLES,
                            search_half_width=SEARCH_HALF_WIDTH,
                            gradient_threshold=GRADIENT_THRESHOLD,
                            radius_tolerance=RADIUS_TOLERANCE):
    gray = preprocess_gray(frame)
    h, w = gray.shape
    cx, cy = float(seed_center[0]), float(seed_center[1])

    raw_edge_points = []
    filtered_edge_points = []

    for k in range(n_angles):
        theta = 2.0 * np.pi * k / n_angles
        if not angle_in_sector(theta, sector):
            continue

        ct = np.cos(theta)
        st = np.sin(theta)

        r_min = max(5, int(seed_radius - search_half_width))
        r_max = int(seed_radius + search_half_width)

        samples = []
        coords = []

        for r in range(r_min, r_max + 1):
            x = cx + r * ct
            y = cy + r * st

            ix = int(round(x))
            iy = int(round(y))

            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                samples.append(0.0)
                coords.append((x, y))
                continue

            samples.append(float(gray[iy, ix]))
            coords.append((x, y))

        if len(samples) < 5:
            continue

        samples = np.array(samples, dtype=np.float32)
        grad = np.abs(np.gradient(samples))

        inner = 2
        if len(grad) <= 2 * inner:
            continue

        grad_inner = grad[inner:-inner]
        idx_local = int(np.argmax(grad_inner)) + inner
        gmax = float(grad[idx_local])

        x_edge, y_edge = coords[idx_local]
        raw_edge_points.append([x_edge, y_edge])

        if gmax < gradient_threshold:
            continue

        dist = np.hypot(x_edge - cx, y_edge - cy)
        if abs(dist - seed_radius) <= radius_tolerance:
            filtered_edge_points.append([x_edge, y_edge])

    if len(filtered_edge_points) < 3:
        return None, None, raw_edge_points, filtered_edge_points, None

    center, radius = fit_circle_least_squares(filtered_edge_points)
    ellipse = fit_ellipse_if_possible(filtered_edge_points)

    return center, radius, raw_edge_points, filtered_edge_points, ellipse


def compute_multi_summary(results):
    if len(results) == 0:
        return None

    centers = np.array([r["center"] for r in results], dtype=np.float32)
    mean_center = np.mean(centers, axis=0)

    d = centers - mean_center
    radial = np.sqrt(np.sum(d ** 2, axis=1))
    rms = float(np.sqrt(np.mean(radial ** 2)))
    max_dev = float(np.max(radial))

    ellipse_ratios = []
    ellipse_centers = []
    n_points_list = []

    for r in results:
        n_points_list.append(r["n_points"])
        ell = r.get("ellipse")
        if ell is not None:
            (ecx, ecy), (a1, a2), _ = ell
            major = max(a1, a2)
            minor = min(a1, a2)
            if major > 0:
                ellipse_ratios.append(minor / major)
                ellipse_centers.append([ecx, ecy])

    mean_ratio = float(np.mean(ellipse_ratios)) if ellipse_ratios else None
    mean_ellipse_center = np.mean(np.array(ellipse_centers, dtype=np.float32), axis=0) if ellipse_centers else None
    mean_n_points = float(np.mean(n_points_list)) if n_points_list else 0.0

    return {
        "mean_center": mean_center,
        "rms": rms,
        "max_dev": max_dev,
        "mean_ratio": mean_ratio,
        "mean_ellipse_center": mean_ellipse_center,
        "mean_n_points": mean_n_points,
    }


def print_multi_summary(results):
    summary = compute_multi_summary(results)
    if summary is None:
        print("No multi-angle measurements available.")
        return

    print("\n=== MULTI-ANGLE SUMMARY ===")
    for r in results:
        msg = (
            f"Angle {r['angle']:>3}° : "
            f"center=({r['center'][0]:.2f}, {r['center'][1]:.2f})  "
            f"radius={r['radius']:.2f}  "
            f"points={r['n_points']}"
        )

        ell = r.get("ellipse")
        if ell is not None:
            (_, _), (a1, a2), _ = ell
            major = max(a1, a2)
            minor = min(a1, a2)
            if major > 0:
                ratio = minor / major
                msg += f"  ellipse_ratio={ratio:.4f}"

        print(msg)

    mc = summary["mean_center"]
    print(f"\nMean center       : ({mc[0]:.2f}, {mc[1]:.2f})")
    print(f"RMS dispersion    : {summary['rms']:.2f} px")
    print(f"Max deviation     : {summary['max_dev']:.2f} px")
    print(f"Mean point count  : {summary['mean_n_points']:.1f}")

    if summary["mean_ratio"] is not None:
        print(f"Mean ellipse ratio: {summary['mean_ratio']:.4f}")

    if summary["mean_ellipse_center"] is not None:
        mec = summary["mean_ellipse_center"]
        print(f"Mean ellipse center: ({mec[0]:.2f}, {mec[1]:.2f})")

    print("===========================\n")


def reset_current_measure():
    global circle_points, sector_points
    global detected_center, detected_radius, detected_edges, filtered_edges, detected_ellipse
    global last_frozen_frame
    global pointing_circle_mode, pointing_sector_mode

    circle_points = []
    sector_points = []
    detected_center = None
    detected_radius = None
    detected_edges = []
    filtered_edges = []
    detected_ellipse = None
    last_frozen_frame = None
    pointing_circle_mode = False
    pointing_sector_mode = False


def extract_zoom(image, center_xy, zoom_factor=2.0, out_size=900):
    h, w = image.shape[:2]
    cx, cy = float(center_xy[0]), float(center_xy[1])

    half_w = int(out_size / (2 * zoom_factor))
    half_h = int(out_size / (2 * zoom_factor))

    x1 = max(0, int(round(cx - half_w)))
    y1 = max(0, int(round(cy - half_h)))
    x2 = min(w, int(round(cx + half_w)))
    y2 = min(h, int(round(cy + half_h)))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return image.copy(), (0, 0), 1.0

    zoomed = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return zoomed, (x1, y1), zoom_factor


def global_to_zoom(pt, origin_xy, zoom_factor):
    x = (pt[0] - origin_xy[0]) * zoom_factor
    y = (pt[1] - origin_xy[1]) * zoom_factor
    return np.array([x, y], dtype=np.float32)


def detect_optical_center_in_roi(frame, expected_center, roi_half_size=90):
    h, w = frame.shape[:2]
    cx, cy = int(expected_center[0]), int(expected_center[1])

    x1 = max(0, cx - roi_half_size)
    y1 = max(0, cy - roi_half_size)
    x2 = min(w, cx + roi_half_size)
    y2 = min(h, cy + roi_half_size)

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    best = None
    best_score = -1e9

    roi_cx = (x2 - x1) / 2.0
    roi_cy = (y2 - y1) / 2.0

    for cnt in contours:
        if len(cnt) < 20:
            continue

        area = cv2.contourArea(cnt)
        if area < 20 or area > 12000:
            continue

        try:
            ell = cv2.fitEllipse(cnt)
        except cv2.error:
            continue

        (ex, ey), (a1, a2), _ = ell
        major = max(a1, a2)
        minor = min(a1, a2)

        if major < 8 or major > 140:
            continue
        if minor < 5:
            continue

        ratio = minor / major
        if ratio < 0.45:
            continue

        dist_to_roi_center = math.hypot(ex - roi_cx, ey - roi_cy)
        score = 2.0 * ratio - 0.015 * dist_to_roi_center - 0.00003 * abs(area - 1200)

        if score > best_score:
            best_score = score
            best = ell

    if best is None:
        return None, None

    (ex, ey), _, _ = best
    global_center = np.array([x1 + ex, y1 + ey], dtype=np.float32)
    return global_center, best


def finalize_review_analysis(summary):
    global analysis_optical_centers

    target_center = summary["mean_center"]

    if len(analysis_optical_centers) == 0:
        return {
            "frames_used": 0,
            "mean_optical_center": None,
            "mean_collimation_error_px": None,
            "std_collimation_error_px": None,
            "collimation_confidence": 0.0,
        }

    optical_centers = np.array(analysis_optical_centers, dtype=np.float32)
    mean_optical_center = np.mean(optical_centers, axis=0)

    offsets = np.sqrt(np.sum((optical_centers - target_center) ** 2, axis=1))
    mean_err = float(np.mean(offsets))
    std_err = float(np.std(offsets))

    s_err = math.exp(- (mean_err / 3.0) ** 2)
    s_stab = math.exp(- (std_err / 2.0) ** 2)
    s_count = np.clip(len(optical_centers) / 20.0, 0.0, 1.0)

    col_score = 100.0 * (0.55 * s_err + 0.25 * s_stab + 0.20 * s_count)

    return {
        "frames_used": len(optical_centers),
        "mean_optical_center": mean_optical_center,
        "mean_collimation_error_px": mean_err,
        "std_collimation_error_px": std_err,
        "collimation_confidence": float(np.clip(col_score, 0.0, 100.0)),
    }


def draw_review_view(frame, summary, review_analysis, analysis_running, analysis_remaining):
    read_review_trackbars()

    center = summary["mean_center"]
    zoomed, origin, zoom_factor = extract_zoom(frame, center, zoom_factor=2.0, out_size=900)

    zc = global_to_zoom(center, origin, zoom_factor)
    draw_full_cross(zoomed, zc, color=(255, 255, 255), thickness=1)

    r1 = max(1, int(circle1_diameter * zoom_factor / 2.0))
    r2 = max(1, int(circle2_diameter * zoom_factor / 2.0))

    cv2.circle(zoomed, (int(zc[0]), int(zc[1])), r1, (255, 255, 0), 1)
    cv2.circle(zoomed, (int(zc[0]), int(zc[1])), r2, (0, 255, 255), 1)

    if review_analysis is not None and review_analysis.get("mean_optical_center") is not None:
        zo = global_to_zoom(review_analysis["mean_optical_center"], origin, zoom_factor)
        cv2.circle(zoomed, (int(zo[0]), int(zo[1])), 5, (0, 0, 255), -1)
        draw_cross(zoomed, zo, (0, 0, 255), size=10, thickness=1)

    cv2.putText(zoomed, f"Review mode x2 - mean center ({center[0]:.1f}, {center[1]:.1f})",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(zoomed, f"Circle 1 diameter: {circle1_diameter} px",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
    cv2.putText(zoomed, f"Circle 2 diameter: {circle2_diameter} px",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    y0 = 130
    if review_analysis is not None:
        err = review_analysis.get("mean_collimation_error_px")
        std = review_analysis.get("std_collimation_error_px")
        colc = review_analysis.get("collimation_confidence")
        used = review_analysis.get("frames_used", 0)

        if err is not None:
            cv2.putText(zoomed, f"Measured collimation error: {err:.2f} px",
                        (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(zoomed, f"Error stability: {std:.2f} px",
                        (20, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(zoomed, f"Collimation confidence: {colc:.1f} / 100",
                        (20, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
            cv2.putText(zoomed, f"Useful frames: {used}",
                        (20, y0 + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)
        else:
            cv2.putText(zoomed, "Measured collimation error: not detected",
                        (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    if analysis_running:
        cv2.putText(zoomed, f"Analysis running... {analysis_remaining:.1f} s",
                    (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)
        cv2.putText(zoomed, "Gently rotate the camera during analysis",
                    (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    cv2.putText(zoomed, "Trackbars: circle diameters | t: 5 s analysis | o: back | ESC: quit",
                (20, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)

    return zoomed


def mouse(event, x, y, flags, param):
    global circle_points, sector_points
    global pointing_circle_mode, pointing_sector_mode, mouse_pos

    mouse_pos = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if pointing_circle_mode:
            if len(circle_points) < 3:
                circle_points.append([x, y])
            if len(circle_points) == 3:
                pointing_circle_mode = False

        elif pointing_sector_mode:
            if len(sector_points) < 3:
                sector_points.append([x, y])
            if len(sector_points) == 3:
                pointing_sector_mode = False


cv2.namedWindow("Camera")
cv2.setMouseCallback("Camera", mouse)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    summary = compute_multi_summary(multi_results)

    if show_review_view and summary is not None:
        ensure_review_trackbars()

        analysis_remaining = 0.0
        if analysis_running:
            elapsed = time.time() - analysis_start_time
            analysis_remaining = max(0.0, ANALYSIS_DURATION_SEC - elapsed)

            optical_center, _ = detect_optical_center_in_roi(frame, summary["mean_center"], roi_half_size=90)
            if optical_center is not None:
                analysis_optical_centers.append(optical_center)

            if elapsed >= ANALYSIS_DURATION_SEC:
                analysis_running = False
                review_analysis = finalize_review_analysis(summary)
                if review_analysis["mean_collimation_error_px"] is not None:
                    print(
                        f"5 s analysis complete: "
                        f"error={review_analysis['mean_collimation_error_px']:.2f} px, "
                        f"stability={review_analysis['std_collimation_error_px']:.2f} px, "
                        f"confidence={review_analysis['collimation_confidence']:.1f}/100, "
                        f"frames={review_analysis['frames_used']}"
                    )
                else:
                    print("5 s analysis complete: optical center not detected.")

        review_img = draw_review_view(frame, summary, review_analysis, analysis_running, analysis_remaining)
        cv2.imshow(REVIEW_WINDOW, review_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord('o'):
            show_review_view = False
            review_analysis = None
            analysis_running = False
            analysis_optical_centers = []
            cv2.destroyWindow(REVIEW_WINDOW)
            trackbars_created = False
        elif key == ord('t'):
            analysis_running = True
            analysis_start_time = time.time()
            analysis_optical_centers = []
            review_analysis = None
            print("5 s analysis started. Gently rotate the camera.")
        continue

    display = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness)

    preview_circle_points = circle_points.copy()
    if pointing_circle_mode and mouse_pos is not None and len(preview_circle_points) < 3:
        preview_circle_points = preview_circle_points + [[mouse_pos[0], mouse_pos[1]]]

    for p in circle_points:
        cv2.circle(display, tuple(p), 6, (0, 255, 0), -1)

    if len(preview_circle_points) == 2:
        cv2.line(display, tuple(preview_circle_points[0]), tuple(preview_circle_points[1]), (0, 255, 255), 1)

    if len(preview_circle_points) == 3:
        center, radius = circle_from_3pts(
            preview_circle_points[0],
            preview_circle_points[1],
            preview_circle_points[2]
        )
        if center is not None:
            cv2.circle(display, tuple(center.astype(int)), int(radius), (255, 0, 0), 2)
            cv2.circle(display, tuple(center.astype(int)), 4, (0, 0, 255), -1)

        cv2.line(display, tuple(preview_circle_points[0]), tuple(preview_circle_points[1]), (0, 255, 255), 1)
        cv2.line(display, tuple(preview_circle_points[1]), tuple(preview_circle_points[2]), (0, 255, 255), 1)
        cv2.line(display, tuple(preview_circle_points[2]), tuple(preview_circle_points[0]), (0, 255, 255), 1)

    seed_center = None
    seed_radius = None
    if len(circle_points) == 3:
        seed_center, seed_radius = circle_from_3pts(circle_points[0], circle_points[1], circle_points[2])
        if seed_center is not None:
            cv2.circle(display, tuple(seed_center.astype(int)), int(seed_radius), (255, 0, 0), 2)
            cv2.circle(display, tuple(seed_center.astype(int)), 4, (0, 0, 255), -1)

    preview_sector_points = sector_points.copy()
    if pointing_sector_mode and mouse_pos is not None and len(preview_sector_points) < 3:
        preview_sector_points = preview_sector_points + [[mouse_pos[0], mouse_pos[1]]]

    for p in sector_points:
        cv2.circle(display, tuple(p), 5, (255, 0, 255), -1)

    sector = None
    if seed_center is not None and len(preview_sector_points) == 3:
        sector = build_sector_from_3pts(seed_center, preview_sector_points)
        if sector is not None:
            draw_sector_arc(display, seed_center, seed_radius, sector, color=(255, 0, 255), thickness=2)

        cv2.line(display, tuple(preview_sector_points[0]), tuple(preview_sector_points[1]), (255, 0, 255), 1)
        cv2.line(display, tuple(preview_sector_points[1]), tuple(preview_sector_points[2]), (255, 0, 255), 1)
        cv2.line(display, tuple(preview_sector_points[2]), tuple(preview_sector_points[0]), (255, 0, 255), 1)

    elif seed_center is not None and len(sector_points) == 3:
        sector = build_sector_from_3pts(seed_center, sector_points)
        if sector is not None:
            draw_sector_arc(display, seed_center, seed_radius, sector, color=(255, 0, 255), thickness=2)

    for p in detected_edges:
        cv2.circle(display, (int(p[0]), int(p[1])), 2, (255, 180, 0), -1)

    for p in filtered_edges:
        cv2.circle(display, (int(p[0]), int(p[1])), 3, (255, 255, 0), -1)

    if detected_center is not None and detected_radius is not None:
        cv2.circle(display, tuple(detected_center.astype(int)), int(detected_radius), (0, 255, 255), 2)
        draw_cross(display, detected_center, (0, 255, 255), size=14, thickness=2)

    if detected_ellipse is not None:
        cv2.ellipse(display, detected_ellipse, (0, 180, 255), 1)

    if summary is not None:
        mean_center = summary["mean_center"]
        draw_cross(display, mean_center, (0, 0, 255), size=18, thickness=2)

    if multi_mode:
        current_angle = multi_angles[multi_index] if multi_index < len(multi_angles) else None
        cv2.putText(display, f"MULTI-ANGLE MODE - next angle: {current_angle} deg",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    elif pointing_circle_mode:
        cv2.putText(display, "CIRCLE MODE: click 3 points", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    elif pointing_sector_mode:
        cv2.putText(display, "SECTOR MODE: click 3 points", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    else:
        cv2.putText(display,
                    "p:circle  u:sector  d:detect  n:save  m:multi-angle  b:review  ESC:quit",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 2)

    cv2.putText(display, "f:freeze  v:reset sector  r:reset all  c:cancel",
                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 220), 2)

    cv2.putText(display,
                f"a/z:brightness {brightness:+d}  s/x:contrast {contrast:.1f}  g/h:gradient {GRADIENT_THRESHOLD:.1f}  j/k:radius tol {RADIUS_TOLERANCE:.1f}",
                (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 2)

    if summary is not None:
        mc = summary["mean_center"]
        txt = f"Mean multi-angle center: ({mc[0]:.1f}, {mc[1]:.1f})  RMS={summary['rms']:.2f}px"
        cv2.putText(display, txt, (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 0, 255), 2)

    cv2.imshow("Camera", display)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break
    elif key == ord('a'):
        brightness -= 5
    elif key == ord('z'):
        brightness += 5
    elif key == ord('s'):
        contrast = max(0.1, contrast - 0.1)
    elif key == ord('x'):
        contrast += 0.1
    elif key == ord('p'):
        pointing_circle_mode = True
        pointing_sector_mode = False
        circle_points = []
        sector_points = []
        detected_center = None
        detected_radius = None
        detected_edges = []
        filtered_edges = []
        detected_ellipse = None
    elif key == ord('u'):
        if len(circle_points) == 3:
            pointing_sector_mode = True
            pointing_circle_mode = False
            sector_points = []
            detected_center = None
            detected_radius = None
            detected_edges = []
            filtered_edges = []
            detected_ellipse = None
    elif key == ord('c'):
        pointing_circle_mode = False
        pointing_sector_mode = False
    elif key == ord('v'):
        sector_points = []
        detected_center = None
        detected_radius = None
        detected_edges = []
        filtered_edges = []
        detected_ellipse = None
    elif key == ord('r'):
        reset_current_measure()
        multi_mode = False
        multi_index = 0
        multi_results = []
        show_review_view = False
        review_analysis = None
        analysis_running = False
        analysis_optical_centers = []
        if trackbars_created:
            cv2.destroyWindow(REVIEW_WINDOW)
            trackbars_created = False
    elif key == ord('f'):
        last_frozen_frame = frame.copy()
        print("Frame frozen.")
    elif key == ord('g'):
        GRADIENT_THRESHOLD = max(1.0, GRADIENT_THRESHOLD - 1.0)
    elif key == ord('h'):
        GRADIENT_THRESHOLD += 1.0
    elif key == ord('j'):
        RADIUS_TOLERANCE = max(2.0, RADIUS_TOLERANCE - 1.0)
    elif key == ord('k'):
        RADIUS_TOLERANCE += 1.0
    elif key == ord('d'):
        if len(circle_points) == 3 and seed_center is not None:
            work_frame = last_frozen_frame if last_frozen_frame is not None else frame.copy()

            current_sector = None
            if len(sector_points) == 3:
                current_sector = build_sector_from_3pts(seed_center, sector_points)

            center, radius, raw_edges, good_edges, ellipse = detect_circle_from_seed(
                work_frame, seed_center, seed_radius, sector=current_sector,
                n_angles=N_ANGLES, search_half_width=SEARCH_HALF_WIDTH,
                gradient_threshold=GRADIENT_THRESHOLD, radius_tolerance=RADIUS_TOLERANCE
            )

            detected_edges = raw_edges
            filtered_edges = good_edges
            detected_ellipse = ellipse

            if center is not None:
                detected_center = center
                detected_radius = radius
                msg = (
                    f"Single-frame detection: x={center[0]:.2f}, y={center[1]:.2f}, "
                    f"radius={radius:.2f}, filtered_points={len(good_edges)}"
                )
                if ellipse is not None:
                    (_, _), (a1, a2), _ = ellipse
                    major = max(a1, a2)
                    minor = min(a1, a2)
                    if major > 0:
                        msg += f", ellipse_ratio={minor/major:.4f}"
                print(msg)
            else:
                detected_center = None
                detected_radius = None
                detected_ellipse = None
                print("Detection failed.")
    elif key == ord('m'):
        multi_mode = True
        multi_index = 0
        multi_results = []
        reset_current_measure()
        show_review_view = False
        review_analysis = None
        analysis_running = False
        analysis_optical_centers = []
        print("\nMulti-angle acquisition started.")
        print("Measure in sequence: 0°, 90°, 180°, 270°")
        print("For each angle: p -> circle, u -> sector, d -> detect, n -> save\n")
    elif key == ord('n'):
        if multi_mode:
            if detected_center is None or detected_radius is None:
                print("No valid detection to save.")
            else:
                angle = multi_angles[multi_index]
                multi_results.append({
                    "angle": angle,
                    "center": detected_center.copy(),
                    "radius": float(detected_radius),
                    "n_points": len(filtered_edges),
                    "ellipse": detected_ellipse,
                })
                print(f"Measurement saved for angle {angle}°.")
                multi_index += 1
                if multi_index >= len(multi_angles):
                    multi_mode = False
                    print_multi_summary(multi_results)
                else:
                    reset_current_measure()
                    print(f"Move to angle {multi_angles[multi_index]}°.")
        else:
            print("Multi-angle mode is not active.")
    elif key == ord('b'):
        print_multi_summary(multi_results)
        if compute_multi_summary(multi_results) is not None:
            show_review_view = True
            review_analysis = None
            analysis_running = False
            analysis_optical_centers = []
            ensure_review_trackbars()

cap.release()
cv2.destroyAllWindows()
