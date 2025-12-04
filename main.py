import cv2
import os
import csv
import numpy as np

# ==== CONFIG ====
VIDEO_PATH = "input/vid1.mp4"
OUTPUT_MINIMAP_VIDEO = "output/minimap_tracked.mp4"
OUTPUT_CSV = "output/tracks.csv"

# From ROI selection script
MINIMAP_ROI = (926, 364, 1266, 705)  # (x_min, y_min, x_max, y_max)

MIN_BLOB_AREA = 20          # min area for a blob to be considered (pixels)
MAX_TRACK_DIST = 25         # max distance (in pixels) to associate a detection to an existing track

# HSV ranges for team colors (you can tweak these if needed)
# Blue team icons
LOWER_BLUE = np.array([90, 50, 50])
UPPER_BLUE = np.array([140, 255, 255])

# Red team icons (wrapped around hue 0)
LOWER_RED1 = np.array([0, 70, 50])
UPPER_RED1 = np.array([10, 255, 255])
LOWER_RED2 = np.array([170, 70, 50])
UPPER_RED2 = np.array([180, 255, 255])


def get_centroids_from_mask(mask, min_area, color_label):
    """Return a list of detections: dicts with x, y, color."""
    detections = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

    for label in range(1, num_labels):  # 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        cx, cy = centroids[label]
        detections.append({
            "x": int(cx),
            "y": int(cy),
            "color": color_label,
        })
    return detections


def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found at {VIDEO_PATH}")
        return

    os.makedirs("output", exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[ERROR] Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    x_min, y_min, x_max, y_max = MINIMAP_ROI
    width = x_max - x_min
    height = y_max - y_min

    print(f"[INFO] FPS: {fps}")
    print(f"[INFO] Frame count: {frame_count}")
    print(f"[INFO] Minimap size: {width} x {height}")

    # Video writer for tracked minimap
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_MINIMAP_VIDEO, fourcc, fps, (width, height))

    # ---- Multi-object tracks ----
    # Each track: {"id": int, "color": "blue"/"red", "last_pos": (x, y), "points": [(x, y), ...]}
    tracks = []
    next_track_id = 0

    # For CSV logging
    track_rows = []

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Crop minimap ----
        minimap = frame[y_min:y_max, x_min:x_max]

        # ---- HSV color thresholding ----
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)

        # Blue team
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        # Red team (two ranges merged)
        mask_red1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
        mask_red2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        # ---- Morphological cleanup ----
        kernel = np.ones((3, 3), np.uint8)
        mask_blue_clean = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
        mask_red_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

        # ---- Get detections for this frame ----
        detections = []
        detections.extend(get_centroids_from_mask(mask_blue_clean, MIN_BLOB_AREA, "blue"))
        detections.extend(get_centroids_from_mask(mask_red_clean, MIN_BLOB_AREA, "red"))

        # ---- Associate detections to existing tracks (greedy nearest-neighbor per color) ----
        for det in detections:
            dx_best = None
            best_track = None
            det_x, det_y, det_color = det["x"], det["y"], det["color"]

            for track in tracks:
                if track["color"] != det_color:
                    continue
                last_x, last_y = track["last_pos"]
                dx = det_x - last_x
                dy = det_y - last_y
                dist2 = dx * dx + dy * dy

                if dist2 <= MAX_TRACK_DIST * MAX_TRACK_DIST:
                    if dx_best is None or dist2 < dx_best:
                        dx_best = dist2
                        best_track = track

            if best_track is None:
                # Start a new track
                track = {
                    "id": next_track_id,
                    "color": det_color,
                    "last_pos": (det_x, det_y),
                    "points": [(det_x, det_y)],
                }
                tracks.append(track)
                track_id = next_track_id
                next_track_id += 1
            else:
                # Append to existing track
                best_track["last_pos"] = (det_x, det_y)
                best_track["points"].append((det_x, det_y))
                track_id = best_track["id"]

            # Log to CSV
            time_seconds = frame_idx / fps if fps > 0 else 0.0
            track_rows.append({
                "frame_index": frame_idx,
                "time_seconds": time_seconds,
                "track_id": track_id,
                "team_color": det_color,
                "x": det_x,
                "y": det_y,
            })

        # ---- Draw all tracks on minimap ----
        for track in tracks:
            pts = track["points"]
            if len(pts) < 2:
                continue

            # Choose color by team
            if track["color"] == "blue":
                line_color = (200, 200, 255)  # light bluish-white
            else:
                line_color = (50, 180, 255)   # orange-ish for red team

            for i in range(1, len(pts)):
                cv2.line(
                    minimap,
                    pts[i - 1],
                    pts[i],
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

        cv2.putText(
            minimap,
            f"Frame: {frame_idx}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        out.write(minimap)

        # Optional live preview
        cv2.imshow("Minimap Tracked", minimap)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # ---- Write CSV with tracks ----
    if track_rows:
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "frame_index",
                    "time_seconds",
                    "track_id",
                    "team_color",
                    "x",
                    "y",
                ],
            )
            writer.writeheader()
            for row in track_rows:
                writer.writerow(row)

        print(f"[DONE] Saved tracks to {OUTPUT_CSV}")
    else:
        print("[WARN] No tracks found; CSV not written.")

    print(f"[DONE] Saved tracked minimap video to {OUTPUT_MINIMAP_VIDEO}")


if __name__ == "__main__":
    main()
