from argparse import ArgumentParser


def get_arguments() -> dict[str, str]:
    """
    Parse command-line arguments.

    Returns:
        A dictionary of arguments.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-i", "--input",
        type=str,
        default="crowd.mp4",
        #required=True,
        help="Path to input video (e.g. crowd.mp4)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="output.mp4",
        help="Path to save annotated video"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="yolo11n.pt",
        help="Path to load YOLO model (or YOLO model name (e.g. yolo11n.pt))"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default="cpu",
        help="Device to use: 'mps', 'cuda', or 'cpu'"
    )
    return vars(parser.parse_args())
