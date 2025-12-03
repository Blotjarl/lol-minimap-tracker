import cv2
import os

VIDEO_PATH = "input/vid1.mp4"

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Video not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Could not read first frame from video.")
        return

    # Show the full frame so you can select the minimap
    print("Drag a box around the minimap, then press ENTER or SPACE. Press C to cancel.")
    roi = cv2.selectROI("Select Minimap ROI", frame, fromCenter=False, showCrosshair=True)

    cv2.destroyAllWindows()

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("No ROI selected.")
        return

    print("\nMinimap ROI selected:")
    print(f"x_min = {x}")
    print(f"y_min = {y}")
    print(f"x_max = {x + w}")
    print(f"y_max = {y + h}")

    # Optional: show just the cropped minimap to confirm
    minimap = frame[y:y+h, x:x+w]
    cv2.imshow("Cropped Minimap", minimap)
    print("Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
