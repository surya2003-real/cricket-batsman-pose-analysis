# Batsman Tracking with Dynamic Decision Region

This project demonstrates a batsman tracking system using **YOLOv8** (or **yolo11s.pt**) for detecting persons. The system dynamically defines a decision region (displayed as a green bounding box) on each video frame to select and lock a target (the batsman) based on the following specifications:

## Specifications

1. **Decision Box for Target Selection:**
   - **Green Decision Region:**  
     - **Horizontal Boundaries:**  
       From **(center_x − 15% of frame_width)** to **(center_x + 15% of frame_width)**.
     - **Vertical Boundaries:**  
       From **10% of frame_height** (top margin) to **(center_y − 10% of frame_height)** initially.
  
2. **Dynamic Decision Region:**
   - The decision box grows in the y-direction over the frames.  
   - The lower boundary of the decision region increases gradually from **(center_y − 10% of frame_height)** until it reaches **center_y**.

3. **Target Locking Criteria:**
   - Until a target is locked, all detections (persons) whose bounding box centers lie within the decision region are collected.
   - If multiple candidates exist, the candidate whose bounding box center is closest to the decision region's center is chosen and its tracking ID is locked as the target.
   
4. **Tolerance Parameter:**
   - The system can tolerate a specified number of consecutive frames in which the target is missing or its bounding box does not satisfy the decision region criteria.
   - If the target is not found for more consecutive frames than specified by the tolerance parameter, the tracking loop terminates.
   - The tolerance can be set via the command line using `--tolerance` (default is 5 frames).

5. **Exit Conditions:**
   - If the locked target is not detected or its complete bounding box moves out of the decision region for more than the allowed tolerance frames, the system exits.

6. **Optional Saving:**
   - The program accepts an optional parameter (`--save`) to write the processed video (with drawn bounding boxes and decision region) to disk.

7. **Model Options:**
   - While the default model is **yolov8s.pt**, you can also use **yolo11s.pt** if desired.  
   - Specify the model path via the `--model` parameter.

## Requirements

- Python 3.11
- [OpenCV](https://opencv.org/) (`pip install opencv-python`)
- [Ultralytics](https://ultralytics.com/) (`pip install ultralytics`)
- NumPy

## Running the Program

Run the script from the command line:

```bash
python tracking.py --input_path path/to/your/video.avi --draw 1 --model yolov8s.pt --tolerance 5 --save output.mp4
