import cv2

from utils import get_arguments
from video_stream import VideoStream
from yolo_model import YOLOModel


class MainApp:
    """
    Main application class that integrates video stream, YOLO model, and display logic.
    """

    def __init__(self, args: dict[str, str]) -> None:
        """
        Initialize the main components of the application.

        Args:
            args: Command-line arguments.
        """
        self.model = YOLOModel(args["model"], args["device"])
        self.capture = VideoStream(args["input"])
        self.output = args.get("output")

    @staticmethod
    def draw_boxes(frame: any, detection) -> any:
        """
        Draw bounding boxes and confidence scores for each detected person on the frame.

        Args:
            frame: The image frame to draw on.
            detection: A single detection result from YOLOModel.process_frame().

        Returns:
            The frame with drawn bounding boxes and labels.
        """
        bboxes = detection.boxes.xyxy.cpu().numpy()
        scores = detection.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), conf in zip(bboxes, scores):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
            label = f"Person: {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)

        return frame

    def run(self) -> None:
        """
        Main loop: capture frames, perform detection, draw boxes, and display.
        """
        while True:
            grabbed, frame = self.capture.read_frame()
            if not grabbed or frame is None:
                break

            results = self.model.process_frame(frame)
            for result in results:
                frame = self.draw_boxes(frame, result)

            cv2.imshow("Object Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = get_arguments()
    app = MainApp(args)
    app.run()
