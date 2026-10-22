from typing import Any
import torch
from .target import XAITarget


class TargetBuilder:
    """
    Build differentiable scalar targets for XAI methods.
    Адаптировано под YOLOv8 (Ultralytics 8.x) и YOLO26.
    """

    @staticmethod
    def build(model, input_data, target, context=None):
        context = context or {}
        if target.task == "classification":
            return TargetBuilder._build_classification(model, input_data, target)
        if target.task == "detection":
            return TargetBuilder._build_detection(model, input_data, target, context)
        raise ValueError(f"Unsupported target task: {target.task}")

    @staticmethod
    def _build_classification(model, input_data, target):
        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Classification XAI requires a torch.Tensor input.")
        output = model(input_data)
        if not isinstance(output, torch.Tensor):
            raise TypeError("Classification model must return a torch.Tensor.")
        if output.ndim != 2:
            raise ValueError("Classification output must have shape [batch, classes].")
        if not (0 <= target.class_id < output.shape[1]):
            raise IndexError(f"Class ID {target.class_id} is out of range.")
        score = output[0, target.class_id]
        if not score.requires_grad:
            raise RuntimeError("Classification target is not differentiable.")
        return score

    @staticmethod
    def _get_detection_network(model):
        wrapper_model = getattr(model, "model", None)
        if isinstance(wrapper_model, torch.nn.Module) and model.__class__.__name__ != "DetectionModel":
            network = wrapper_model
        else:
            network = model
        if not isinstance(network, torch.nn.Module):
            raise TypeError("Could not resolve a PyTorch detection model.")
        return network

    @staticmethod
    def _box_iou(boxes1, boxes2):
        if boxes1.ndim != 2 or boxes1.shape[1] != 4:
            raise ValueError("boxes1 must have shape [N, 4].")
        if boxes2.ndim != 2 or boxes2.shape[1] != 4:
            raise ValueError("boxes2 must have shape [M, 4].")
        top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = torch.clamp(bottom_right - top_left, min=0.0)
        intersection = (wh[..., 0] * wh[..., 1])
        area1 = (torch.clamp(boxes1[:, 2] - boxes1[:, 0], min=0.0) * torch.clamp(boxes1[:, 3] - boxes1[:, 1], min=0.0))
        area2 = (torch.clamp(boxes2[:, 2] - boxes2[:, 0], min=0.0) * torch.clamp(boxes2[:, 3] - boxes2[:, 1], min=0.0))
        union = (area1[:, None] + area2[None, :] - intersection + 1e-8)
        return (intersection / union)

    @staticmethod
    def _build_detection(model, input_data, target, context=None):
        context = context or {}
        if not isinstance(input_data, torch.Tensor):
            raise TypeError("Detection XAI requires a torch.Tensor input.")
        if input_data.ndim != 4:
            raise ValueError("Detection XAI input must have shape [batch, channels, height, width].")
        if target.bbox is None:
            raise ValueError("Detection XAI requires target.bbox.")

        network = TargetBuilder._get_detection_network(model)
        network.eval()

        if not input_data.requires_grad:
            input_data = input_data.detach().clone().requires_grad_(True)

        with torch.inference_mode(False), torch.enable_grad():
            raw_output = network(input_data)

            # --- АДАПТАЦИЯ ПОД YOLOv8 ---
            y = None
            if isinstance(raw_output, tuple) and len(raw_output) == 2:
                y = raw_output[0]  # YOLOv8 возвращает (y, x)
            elif isinstance(raw_output, torch.Tensor):
                y = raw_output
            elif isinstance(raw_output, list):
                # Список фичей, пропускаем через голову Detect
                network_detect = network.model[-1]
                y = network_detect._inference(raw_output)
                if isinstance(y, tuple):
                    y = y[0]
            else:
                # Проверка на формат YOLO26 (на всякий случай)
                if isinstance(raw_output, tuple) and len(raw_output) == 2 and isinstance(raw_output[1], dict):
                    predictions = raw_output[1]
                    one2one = predictions.get("one2one")
                    if isinstance(one2one, dict):
                        raw_scores = one2one.get("scores")
                        # Если это YOLO26, оставляем старую логику (для совместимости)
                        network_detect = network.model[-1]
                        decoded_boxes = network_detect._get_decode_boxes(one2one)
                        # ... (вставьте старую логику YOLO26, если она нужна)
                        pass
                if y is None:
                    raise ValueError(f"Unsupported YOLO output type: {type(raw_output)}")

            # Формат y: [1, 4+nc, num_anchors]
            if y.ndim != 3:
                raise ValueError(f"YOLO output must have shape [batch, 4+nc, num_anchors], got {y.shape}")

            boxes_xywh = y[0, :4, :].T  # [num_anchors, 4]
            cls_scores = y[0, 4:, :]    # [nc, num_anchors]

            # Конвертация xywh -> xyxy
            boxes_xyxy = torch.zeros_like(boxes_xywh)
            boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
            boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
            boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
            boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

            letterbox = context.get("letterbox")
            if letterbox is None:
                raise ValueError("Detection XAI requires letterbox metadata.")
            scale = float(letterbox["scale"])
            pad_x = float(letterbox["pad_x"])
            pad_y = float(letterbox["pad_y"])
            x1, y1, x2, y2 = target.bbox

            target_box = torch.tensor([
                x1 * scale + pad_x,
                y1 * scale + pad_y,
                x2 * scale + pad_x,
                y2 * scale + pad_y,
            ], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device).unsqueeze(0)

            ious = TargetBuilder._box_iou(boxes_xyxy, target_box)[:, 0]
            class_scores = cls_scores[target.class_id]

            iou_threshold = float(context.get("detection_iou_threshold", 0.10))
            score_threshold = float(context.get("detection_score_threshold", 0.01))
            top_k = int(context.get("detection_top_k", 16))

            candidate_mask = (ious >= iou_threshold) & (class_scores >= score_threshold)
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).flatten()

            if candidate_indices.numel() == 0:
                top_count = min(top_k, ious.numel())
                candidate_indices = torch.topk(ious, k=top_count, largest=True).indices
            else:
                relevance = ious[candidate_indices].detach() * class_scores[candidate_indices].detach()
                top_count = min(top_k, candidate_indices.numel())
                selected = torch.topk(relevance, k=top_count, largest=True).indices
                candidate_indices = candidate_indices[selected]

            selected_ious = ious[candidate_indices]
            selected_scores = class_scores[candidate_indices]
            
            # В YOLOv8 скоры уже после сигмоиды, поэтому берем логит для стабильности
            selected_logits = torch.logit(selected_scores + 1e-8)

            weights = selected_ious.detach() * selected_scores.detach()
            if weights.sum() <= 1e-8:
                weights = torch.ones_like(weights)
            weights = weights / weights.sum()

            target_score = (weights * selected_logits).sum()
            return target_score