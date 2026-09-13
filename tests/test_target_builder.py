from typing import Any

import torch

from uav_xai.core.prediction import (
    DetectionResult,
    PredictionResult,
)
from uav_xai.core.target import (
    XAITarget,
)
from uav_xai.core.target_builder import (
    TargetBuilder,
)


class TargetBuilder:
    """
    Build a differentiable scalar target for XAI.
    """

    @staticmethod
    def build(
        model: Any,
        input_data: Any,
        target: XAITarget,
    ) -> torch.Tensor:

        if target.task == "classification":
            return TargetBuilder._classification(
                model,
                input_data,
                target,
            )

        if target.task == "detection":
            return TargetBuilder._detection(
                model,
                input_data,
                target,
            )

        raise ValueError(
            f"Unsupported target task: "
            f"{target.task}"
        )

    @staticmethod
    def _classification(
        model: Any,
        input_data: torch.Tensor,
        target: XAITarget,
    ) -> torch.Tensor:

        output = model(
            input_data
        )

        if not isinstance(
            output,
            torch.Tensor,
        ):
            raise TypeError(
                "Classification model must return "
                "torch.Tensor."
            )

        if output.ndim != 2:
            raise ValueError(
                "Classification output must have "
                "shape [batch, classes]."
            )

        return output[
            0,
            target.class_id,
        ]

    @staticmethod
    def _detection(
        model: Any,
        input_data: torch.Tensor,
        target: XAITarget,
    ) -> torch.Tensor:
        """
        Build target corresponding exactly to the
        final YOLO end-to-end detection index.

        The YOLO26 Detect head internally produces
        predictions over 8400 positions. Its postprocess()
        selects the final detections with top-k. We reproduce
        that same differentiable path and select the score
        of the requested final detection.
        """

        if not hasattr(
            model,
            "model",
        ):
            raise TypeError(
                "YOLO detection model must expose "
                "the internal model."
            )

        network = model.model

        raw_output = network(
            input_data
        )

        if not (
            isinstance(
                raw_output,
                tuple,
            )
            and len(raw_output) == 2
        ):
            raise ValueError(
                "Expected YOLO raw output "
                "(predictions, auxiliary)."
            )

        _, auxiliary = raw_output

        if not isinstance(
            auxiliary,
            dict,
        ):
            raise ValueError(
                "Expected YOLO auxiliary output dict."
            )

        one2one = auxiliary.get(
            "one2one"
        )

        if one2one is None:
            raise ValueError(
                "YOLO output does not contain "
                "one2one predictions."
            )

        detection_head = network.model[-1]

        if not hasattr(
            detection_head,
            "_inference",
        ):
            raise TypeError(
                "YOLO detection head does not expose "
                "_inference()."
            )

        if not hasattr(
            detection_head,
            "postprocess",
        ):
            raise TypeError(
                "YOLO detection head does not expose "
                "postprocess()."
            )

        decoded = (
            detection_head._inference(
                one2one
            )
        )

        # Detect.postprocess expects:
        # [batch, positions, 4 + classes]
        processed = (
            detection_head.postprocess(
                decoded.permute(
                    0,
                    2,
                    1,
                )
            )
        )

        detection_index = (
            target.detection_index
            if target.detection_index is not None
            else 0
        )

        if not (
            0 <= detection_index
            < processed.shape[1]
        ):
            raise IndexError(
                f"Detection index "
                f"{detection_index} is out of range."
            )

        # Final YOLO format:
        # [x1, y1, x2, y2, confidence, class_id]
        #
        # Column 4 is differentiable confidence.
        score = processed[
            0,
            detection_index,
            4,
        ]

        if not score.requires_grad:
            raise RuntimeError(
                "YOLO detection target is not "
                "differentiable."
            )

        return score