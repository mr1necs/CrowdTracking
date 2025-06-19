import logging
from typing import Optional, Tuple

import cv2


class VideoStream:
    """
    Manages a video stream: opening, reading frames, and releasing the stream.
    """

    def __init__(self, video_path: Optional[str] = None) -> None:
        """
        Initialize the video stream.

        Args:
            video_path (Optional[str]): Path to a video file. If None, uses the default camera.
        """
        self.capture = self._open_stream(video_path)

    def _open_stream(self, video_path: Optional[str]) -> cv2.VideoCapture:
        """
        Open the video stream.

        Args:
            video_path (Optional[str]): Path to a video file. If None, uses the default camera.

        Returns:
            cv2.VideoCapture: Opened video capture object.
        """
        source = 0 if video_path is None else video_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logging.error("Failed to open video source: %s", source)
            exit(1)
        return cap

    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """
        Read the next frame from the video stream.

        Returns:
            Tuple[bool, Optional[any]]: (success, frame), where success indicates if the frame was
            captured, and frame is the image or None.
        """
        success, frame = self.capture.read()
        if not success:
            logging.info("End of stream or failed to capture frame")
            return success, None
        return True, frame

    def release(self) -> None:
        """
        Release the video stream and free resources.
        """
        if self.capture.isOpened():
            self.capture.release()
            logging.info("Video stream released")
