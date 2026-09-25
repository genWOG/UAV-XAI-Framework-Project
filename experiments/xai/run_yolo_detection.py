from pathlib import Path

from ultralytics import YOLO


IMAGE_PATH = Path(
    "data/images/test.jpg"
)

OUTPUT_DIR = Path(
    "results/yolo_detection"
)

MODEL_NAME = "yolo26n.pt"


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("YOLO DETECTION TEST")
    print("=" * 60)

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Image: {IMAGE_PATH}"
    )

    model = YOLO(
        MODEL_NAME
    )

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=0.25,
        verbose=False,
    )

    if not results:
        raise RuntimeError(
            "YOLO returned no results."
        )

    result = results[0]

    print()
    print(
        f"Image size: "
        f"{result.orig_shape[::-1]}"
    )

    print(
        f"Detections: "
        f"{len(result.boxes)}"
    )

    for index, box in enumerate(
        result.boxes,
        start=1,
    ):
        class_id = int(
            box.cls.item()
        )

        confidence = float(
            box.conf.item()
        )

        class_name = model.names[
            class_id
        ]

        xyxy = [
            float(value)
            for value in box.xyxy[
                0
            ].tolist()
        ]

        print(
            f"  [{index}] "
            f"{class_name} "
            f"conf={confidence:.4f} "
            f"box={xyxy}"
        )

    output_path = (
        OUTPUT_DIR
        / "detections.jpg"
    )

    result.save(
        filename=str(
            output_path
        )
    )

    print()
    print(
        f"Saved: {output_path}"
    )

    print("=" * 60)
    print("YOLO DETECTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()