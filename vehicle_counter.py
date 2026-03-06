# Vehicle Counter using YOLOv8 and SORT Tracker
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
import cvzone
import math
from sort import Sort  # Make sure sort.py is in the same folder

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Vehicle Counter")
    parser.add_argument('--video', type=str, default='cars.mp4',
                        help='Path to input video (use 0 for webcam)')
    parser.add_argument('--mask', type=str, default='mask.png',
                        help='Path to ROI mask image')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save output video (e.g. output/counting.avi)')
    parser.add_argument('--line', nargs=4, type=int, default=[400, 297, 673, 297],
                        help='Counting line coordinates: x1 y1 x2 y2')
    args = parser.parse_args()

    # Load video
    cap = cv2.VideoCapture(args.video if args.video != '0' else 0)
    if not cap.isOpened():
        print("Error: Could not open video source")
        return

    # Load YOLO model (auto-downloads yolov8n.pt on first run)
    model = YOLO("yolov8n.pt")
    classNames = model.names

    # Load mask
    mask = cv2.imread(args.mask)
    if mask is None:
        print(f"Error: Mask file '{args.mask}' not found!")
        return

    # Tracker
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Counting variables
    totalCounts = set()                     # Using set for O(1) lookup
    limits = args.line

    # Output video writer (if requested)
    out = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f" Saving output to: {args.output}")

    print("Vehicle Counter Started! Press 'q' to quit.")

    while True:
        success, img = cap.read()
        if not success:
            print("End of video reached.")
            break

        # Apply ROI mask
        imgRegion = cv2.bitwise_and(img, mask)

        # YOLO Detection
        results = model(imgRegion, stream=True)
        detections = np.empty((0, 5))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass in ["car", "bus", "truck", "motorbike"] and conf > 0.3:
                    detections = np.vstack((detections, [x1, y1, x2, y2, conf]))

        # Update tracker
        resultsTracker = tracker.update(detections)

        # Draw counting line
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2

            # Draw tracking box and ID
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'ID: {id}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Count vehicle crossing line
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if id not in totalCounts:
                    totalCounts.add(id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # Display count
        cvzone.putTextRect(img, f'VEHICLES: {len(totalCounts)}', (50, 50), scale=2, thickness=3, offset=10)

        # FPS
        fps = cap.get(cv2.CAP_PROP_FPS)
        cv2.putText(img, f'FPS: {int(fps)}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Write frame if saving
        if out is not None:
            out.write(img)

        cv2.imshow("YOLOv8 Vehicle Counter", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()