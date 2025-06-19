import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from tqdm import tqdm

from modules.utils import get_arguments
from modules.video_stream import VideoStream
from modules.yolo_model import YOLOModel


class MainApp:
    """
    Main application for video streaming, YOLO object detection,
    and display logic integration.
    """

    def __init__(self, args: Dict[str, str]) -> None:
        """
        Initialize the MainApp with model, video stream, and output settings.

        Args:
            args (Dict[str, str]): Command-line arguments with keys:
                - "model": Path to the YOLO model file.
                - "device": Device to run inference on ("cpu", "cuda", "mps").
                - "input": Path to video file or None for camera.
                - "output": Optional path to save output video.
                - "show": Whether to display the video.
        """
        self.model = YOLOModel(args["model"], args["device"])
        self.video_stream = VideoStream(args["input"])
        self.output_path: Optional[str] = args.get("output")
        self.show = args.get("show")

        if self.output_path:
            fourcc = cv2.VideoWriter.fourcc(*"mp4v")
            fps = int(self.video_stream.capture.get(cv2.CAP_PROP_FPS)) or 30
            width = int(self.video_stream.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.video_stream.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(
                self.output_path, fourcc, fps, (width, height)
            )
        else:
            self.video_writer = None

    @staticmethod
    def draw_boxes(frame: np.ndarray, detections: Any) -> np.ndarray:
        """
        Draw bounding boxes and confidence labels on the frame.

        Args:
            frame (np.ndarray): Image on which to draw.
            detections (Any): Result from YOLOModel.process_frame().

        Returns:
            np.ndarray: The annotated image.
        """
        for det in detections:
            bboxes = det.boxes.xyxy.cpu().numpy()
            scores = det.boxes.conf.cpu().numpy()

            for (x1, y1, x2, y2), conf in zip(bboxes, scores):
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
                label = f"Person: {conf:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def _cleanup(self) -> None:
        """
        Release resources and close all OpenCV windows.
        """
        self.video_stream.release()
        if self.video_writer:
            self.video_writer.release()
            logging.info(f"Output saved to {self.output_path}")
        if self.show:
            cv2.destroyAllWindows()

    def run(self) -> None:
        """
        Run the main loop: read frames, detect objects, annotate, display, and save.
        """
        total_frames = int(self.video_stream.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pbar = tqdm(total=total_frames, desc="Processing video", unit="frame")

        while True:
            grabbed, frame = self.video_stream.read_frame()
            if not grabbed or frame is None:
                logging.info("No more frames or failed to capture.")
                break

            results = self.model.process_frame(frame)
            annotated = self.draw_boxes(frame, results)

            if self.show:
                cv2.imshow("Object Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logging.info("Exit key pressed.")
                    break

            if self.video_writer:
                self.video_writer.write(annotated)

            pbar.update(1)

        pbar.close()
        self._cleanup()


def main():
    """
    Entry point for the application. Parses arguments, initializes the app, and runs it.
    """
    args = get_arguments()
    app = MainApp(args)
    app.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
    )
    main()
