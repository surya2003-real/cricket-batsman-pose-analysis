import cv2
import numpy as np
import argparse
from ultralytics import YOLO

# Helper function to check if therse is no intersection between two boxes.
def no_intersection(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    # Check if the box is completely outside the decision region.
    if (x2 < xmin or x1 > xmax or y2 < ymin or y1 > ymax):
        return True
    # Check if the box is completely inside the decision region.
    if (x1 >= xmin and x2 <= xmax and y1 >= ymin and y2 <= ymax):
        return False
    # Otherwise, check for partial intersection.
    return not (x1 < xmax and x2 > xmin and y1 < ymax and y2 > ymin)
# Helper function to verify if at least one corner of the box is within the decision region.
def any_corner_inside(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    for (x, y) in corners:
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True
    return False

# Helper function to check if the center of a box is inside a region.
def is_center_inside(x1, y1, x2, y2, xmin, ymin, xmax, ymax):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    return (xmin <= cx <= xmax) and (ymin <= cy <= ymax)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='test_vid_4.avi', help='source video')
parser.add_argument('--draw', type=int, default=1, help='draw bounding boxes (1 to enable)')
parser.add_argument('--model', type=str, default='yolov8s.pt', help='YOLOv8 model to use')
parser.add_argument('--save', type=str, default='', 
                    help='Optional path to save output video. Leave empty for no saving')
# New optional parameter: tolerance (number of frames to wait after disappearance)
parser.add_argument('--tolerance', type=int, default=5, help='Number of consecutive frames to wait after target disappearance')
args = parser.parse_args()

# Load YOLOv8 model.
model = YOLO(args.model)

# Open video capture.
cap = cv2.VideoCapture(args.input_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read one frame to get frame dimensions.
ret, frame = cap.read()
if not ret:
    print("Error: Could not read a frame from the video.")
    exit()

frame_height, frame_width, _ = frame.shape
center_x, center_y = frame_width // 2, frame_height // 2

# Dynamically calculate margins for the "upper central" region.
x_margin = int(frame_width * 0.15)         # 15% of the frame width.
y_margin_top = int(frame_height * 0.1)       # 10% of the frame height from the top.
y_margin_bottom = int(frame_height * 0.1)    # 10% of the frame height from the center downward.

upper_central_xmin = center_x - x_margin
upper_central_xmax = center_x + x_margin
upper_central_ymin = y_margin_top

# The initial lower boundary of the decision box.
initial_upper_central_ymax = center_y - y_margin_bottom

# Total number of frames over which the decision box moves until its lower boundary reaches center_y.
T = 40 
# Calculate per-frame step for the lower boundary.
step = (center_y - initial_upper_central_ymax) / T

print("Initial decision box:")
print("  X:", upper_central_xmin, "to", upper_central_xmax)
print("  Y:", upper_central_ymin, "to", initial_upper_central_ymax)

# Reset video capture to start from the first frame.
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Initialize VideoWriter if saving is requested.
video_writer = None
if args.save:
    # Get FPS from the capture, or default to 20.0 if unavailable.
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 20.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.save, fourcc, fps, (frame_width, frame_height))
    print(f"Output video will be saved to: {args.save}")

# To store tracking results.
saved_coordinates = []
bbox_arr = []
frame_idx = 0

# Variable to store the target batsman's track ID once detected.
target_id = None
# Counter for consecutive frames where the target is missing.
lost_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Dynamically update the lower boundary of the decision box.
    dynamic_ymax = initial_upper_central_ymax + step * frame_idx
    if dynamic_ymax > center_y:
        dynamic_ymax = center_y

    # Draw the dynamic decision region on the frame (green rectangle).
    cv2.rectangle(frame,
                  (upper_central_xmin, upper_central_ymin),
                  (upper_central_xmax, int(dynamic_ymax)),
                  (0, 255, 0), 2)

    # Run tracking on the frame (limit to 'person' detections via class 0).
    results = model.track(frame, persist=True, classes=[0])
    current_target_found = False

    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []

        if target_id is None:
            # Collect candidate detections whose center lies within the current decision region.
            candidates = []
            region_center_x = (upper_central_xmin + upper_central_xmax) / 2
            region_center_y = (upper_central_ymin + dynamic_ymax) / 2
            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                if is_center_inside(x1, y1, x2, y2, upper_central_xmin, upper_central_ymin, upper_central_xmax, int(dynamic_ymax)):
                    # Compute the distance between box center and decision-region center.
                    box_cx = (x1 + x2) / 2
                    box_cy = (y1 + y2) / 2
                    distance = np.sqrt((box_cx - region_center_x)**2 + (box_cy - region_center_y)**2)
                    candidates.append((distance, int(track_id), (x1, y1, x2, y2)))
            if candidates:
                # Select candidate with the smallest distance.
                candidates.sort(key=lambda x: x[0])
                chosen = candidates[0]
                target_id = chosen[1]
                x1, y1, x2, y2 = chosen[2]
                print(f"Batsman detected in frame {frame_idx+1} with ID {target_id}")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {target_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                saved_coordinates.append((frame_idx, [x1, y1, x2, y2]))
                bbox_arr.append([x1, y1, x2, y2])
                current_target_found = True
            else:
                print(f"No candidates found in frame {frame_idx+1}.")
                bbox_arr.append([-1, -1, -1, -1])
        else:
            # Target is already locked: process only detections matching target_id.
            for box, track_id in zip(boxes, ids):
                if int(track_id) == target_id:
                    x1, y1, x2, y2 = map(int, box)
                    # Ensure that at least one corner lies inside the dynamic decision region.
                    if no_intersection(x1, y1, x2, y2, upper_central_xmin, upper_central_ymin, upper_central_xmax, int(dynamic_ymax)):
                        print(f"Batsman with ID {target_id} has no corner in the decision box in frame {frame_idx+1}.")
                        current_target_found = False
                        break
                    else:
                        current_target_found = True
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(frame, f"ID: {target_id}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        saved_coordinates.append((frame_idx, [x1, y1, x2, y2]))
                        bbox_arr.append([x1, y1, x2, y2])
                        break
    else:
        # If no valid target is found, append invalid coordinates.
        bbox_arr.append([-1, -1, -1, -1])
        print(f"No valid target found in frame {frame_idx+1}.")
    # Check tolerance if target is locked.
    if target_id is not None:
        if current_target_found:
            lost_frames = 0  # Reset if target is detected.
        else:
            lost_frames += 1
            bbox_arr.append([-1, -1, -1, -1])
            print(f"Target ID {target_id} missing in frame {frame_idx+1} (lost count: {lost_frames}).")
            # Exit if lost count exceeds tolerance.
            if lost_frames >= args.tolerance:
                print(f"Target ID {target_id} missing for {lost_frames} consecutive frames. Exiting tracking loop.")
                break

    # Display the frame in a live video window if drawing is enabled.
    if args.draw:
        cv2.imshow("Live Video - Tracking", frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    # Write the frame to file if saving is enabled.
    if video_writer:
        video_writer.write(frame)

    frame_idx += 1

# Cleanup windows.
cv2.destroyAllWindows()

# Save tracked bounding box coordinates to a text file and NumPy array.
output_file = "filtered_boxes_coordinates.txt"
with open(output_file, "w") as f:
    for idx, box in saved_coordinates:
        f.write(f"Frame {idx + 1}: {box}\n")
print(f"Saved bounding box coordinates to {output_file}")
np.savetxt('rect.npy', np.array(bbox_arr))

# Release video capture and writer resources.
cap.release()
if video_writer:
    video_writer.release()
