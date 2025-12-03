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

# Thresholds / params
DIFF_THRESH = 25          # for frame differencing (0-255)
MIN_BLOB_AREA = 20        # min area for a blob to be considered (pixels)

# HSV ranges for player blue icon (you may need to tweak these)
# These are generic blue ranges – we might refine later if needed.
LOWER_BLUE = (100, 80, 80)
UPPER_BLUE = (130, 255, 255)


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

    # For frame differencing
    prev_gray_minimap = None

    # For storing track info
    player_points = []  # list of (x, y) tuples in minimap coords
    track_rows = []     # rows for CSV: frame_index, time_seconds, track_id, x, y

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # ---- Crop minimap ----
        minimap = frame[y_min:y_max, x_min:x_max]

        # ---- Convert to grayscale for frame differencing ----
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)

        if prev_gray_minimap is None:
            prev_gray_minimap = gray
            # we still write the frame but we cannot compute motion yet
            out.write(minimap)
            frame_idx += 1
            continue

        # ---- Frame differencing ----
        diff = cv2.absdiff(gray, prev_gray_minimap)
        _, diff_thresh = cv2.threshold(diff, DIFF_THRESH, 255, cv2.THRESH_BINARY)

        # update previous frame
        prev_gray_minimap = gray

        # ---- Color thresholding in HSV for blue icon ----
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        mask_blue = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        # ---- Combine motion + color ----
        moving_blue = cv2.bitwise_and(mask_blue, diff_thresh)

        # ---- Morphological cleanup ----
        kernel = np.ones((3, 3), np.uint8)
        moving_blue_clean = cv2.morphologyEx(moving_blue, cv2.MORPH_OPEN, kernel)

        # ---- Connected components to find blobs ----
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(moving_blue_clean)

        player_centroid = None

        # label 0 is background; start at 1
        best_area = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < MIN_BLOB_AREA:
                continue

            cx, cy = centroids[label]
            # choose largest blob as player (simple heuristic)
            if area > best_area:
                best_area = area
                player_centroid = (int(cx), int(cy))

        # ---- Save track data if we found the player ----
        if player_centroid is not None:
            player_points.append(player_centroid)
            time_seconds = frame_idx / fps if fps > 0 else 0.0

            track_rows.append({
                "frame_index": frame_idx,
                "time_seconds": time_seconds,
                "track_id": "player",
                "x": player_centroid[0],
                "y": player_centroid[1],
            })

        # ---- Draw trajectory on minimap ----
        # Draw all previous points as a path
        for i in range(1, len(player_points)):
            cv2.line(
                minimap,
                player_points[i - 1],
                player_points[i],
                (255, 255, 255),  # white line for path
                2,
                cv2.LINE_AA,
            )

        # Draw current position as small circle
        if player_centroid is not None:
            cv2.circle(minimap, player_centroid, 4, (0, 255, 255), -1)  # yellow dot

        # Optional: overlay frame index
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

        # ---- Write minimap frame with drawn path ----
        out.write(minimap)

        # Optional: live preview
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
            writer = csv.DictWriter(f, fieldnames=["frame_index", "time_seconds", "track_id", "x", "y"])
            writer.writeheader()
            for row in track_rows:
                writer.writerow(row)
        print(f"[DONE] Saved tracks to {OUTPUT_CSV}")
    else:
        print("[WARN] No player tracks found; CSV not written.")

    print(f"[DONE] Saved tracked minimap video to {OUTPUT_MINIMAP_VIDEO}")


if __name__ == "__main__":
    main()
