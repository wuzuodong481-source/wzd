#!/usr/bin/env python3
"""辉羲 (Huixi) RPU 芯片推理模块

使用 Rhino (辉羲) NPU 芯片加速 YOLO 目标检测推理。
通过 batch_inference 工具调用 .ref 模型进行推理。

模型 I/O 格式 (best_new.ref):
  输入: images  [1, 3, 640, 640] float32 (NCHW, 0~1 归一化)
  输出: output0 [1, 6, 8400]    float32 (YOLOv8: cx,cy,w,h,cls_a,cls_b)
"""

import os
import time
import shutil
import subprocess
import numpy as np
import cv2

# ── 路径配置 ──
BATCH_INFER_BIN = "/home/agi/app/bin/examples/batch_inference"
DEFAULT_REF_MODEL = "/data/wzd/best_new.ref"

# ── 模型参数 ──
IMGSZ = 640
NUM_CLASSES = 2
CLASS_NAMES = {0: "a", 1: "b"}
OUTPUT_DIM = 4 + NUM_CLASSES  # 6: cx, cy, w, h, cls_a, cls_b
NUM_ANCHORS = 8400

# ── 临时目录 (使用 /dev/shm 加速文件 I/O) ──
_BATCH_DIR = "/dev/shm/rhino_batch"
_OUTPUT_DIR = "/dev/shm/rhino_output"


def _letterbox(img, new_shape=IMGSZ):
    """YOLO letterbox 预处理: 等比缩放 + 居中填充

    Returns:
        img_nchw: [1, 3, H, W] float32 (0~1, RGB, NCHW)
        info: (ratio, pad_left, pad_top) 用于还原坐标
    """
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(h * r), int(w * r)
    img_r = cv2.resize(img, (nw, nh))

    pad_h = new_shape - nh
    pad_w = new_shape - nw
    top, left = pad_h // 2, pad_w // 2
    img_p = cv2.copyMakeBorder(img_r, top, pad_h - top, left, pad_w - left,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))

    # BGR → RGB, normalize, HWC → CHW → NCHW
    img_p = cv2.cvtColor(img_p, cv2.COLOR_BGR2RGB)
    img_p = img_p.astype(np.float32) / 255.0
    img_p = img_p.transpose(2, 0, 1)[np.newaxis, ...]  # [1, 3, 640, 640]
    return img_p, (r, left, top)


def _nms(boxes, scores, iou_threshold=0.45):
    """单类 NMS (NumPy 实现)"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


def postprocess(output, orig_shape, letterbox_info, conf_threshold=0.25,
                iou_threshold=0.45):
    """YOLOv8 后处理: 解析输出张量 → 检测框列表

    Args:
        output: [1, 6, 8400] float32
        orig_shape: (h, w) 原始图片尺寸
        letterbox_info: (ratio, pad_left, pad_top)
        conf_threshold: 置信度阈值
        iou_threshold: NMS IoU 阈值

    Returns:
        list of dict: [{cls, conf, x1, y1, x2, y2, cx, cy}, ...]
    """
    # [1, 6, 8400] → [8400, 6]
    pred = output[0].T  # [8400, 6]

    # 提取 bbox 和类别分数
    boxes_xywh = pred[:, :4]  # cx, cy, w, h
    cls_scores = pred[:, 4:]  # [8400, 2]

    # 每个anchor的最大类别分数和类别索引
    max_scores = cls_scores.max(axis=1)  # [8400]
    max_cls_ids = cls_scores.argmax(axis=1)  # [8400]

    # 置信度过滤
    mask = max_scores >= conf_threshold
    if mask.sum() == 0:
        return []

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    max_cls_ids = max_cls_ids[mask]

    # xywh → xyxy
    cx, cy, w, h = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    # 还原到原始图片坐标 (撤销 letterbox)
    ratio, pad_left, pad_top = letterbox_info
    boxes_xyxy[:, 0] = (boxes_xyxy[:, 0] - pad_left) / ratio
    boxes_xyxy[:, 1] = (boxes_xyxy[:, 1] - pad_top) / ratio
    boxes_xyxy[:, 2] = (boxes_xyxy[:, 2] - pad_left) / ratio
    boxes_xyxy[:, 3] = (boxes_xyxy[:, 3] - pad_top) / ratio

    # 裁剪到图片范围内
    orig_h, orig_w = orig_shape[:2]
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

    # 按类别分组 NMS
    results = []
    for cls_id in range(NUM_CLASSES):
        cls_mask = max_cls_ids == cls_id
        if cls_mask.sum() == 0:
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores_f = max_scores[cls_mask]
        keep = _nms(cls_boxes, cls_scores_f, iou_threshold)
        for idx in keep:
            bx1, by1, bx2, by2 = cls_boxes[idx]
            results.append({
                "cls": CLASS_NAMES[cls_id],
                "conf": float(cls_scores_f[idx]),
                "x1": float(bx1), "y1": float(by1),
                "x2": float(bx2), "y2": float(by2),
                "cx": float((bx1 + bx2) / 2),
                "cy": float((by1 + by2) / 2),
            })
    return results


def draw_annotated(img_bgr, detections):
    """绘制检测框标注图"""
    annotated = img_bgr.copy()
    color_map = {"a": (0, 255, 0), "b": (0, 165, 255)}
    for det in detections:
        cls = det["cls"]
        color = color_map.get(cls, (255, 0, 0))
        x1, y1 = int(det["x1"]), int(det["y1"])
        x2, y2 = int(det["x2"]), int(det["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {det['conf']:.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


class RhinoInfer:
    """辉羲 RPU 芯片 YOLO 推理引擎

    使用 batch_inference 工具进行推理，每次调用会重新加载模型。
    实测总耗时 ~130ms (含模型加载 ~63ms)，远快于远程MQTT推理和本地CPU推理。

    用法:
        engine = RhinoInfer("/data/wzd/best_new.ref")
        detections = engine.infer(img_bgr, conf_threshold=0.25)
        # detections: [{"cls": "a", "conf": 0.95, "x1": ..., "y1": ..., "x2": ..., "y2": ..., "cx": ..., "cy": ...}, ...]
    """

    def __init__(self, ref_model_path=DEFAULT_REF_MODEL):
        self.ref_model = ref_model_path
        if not os.path.isfile(ref_model_path):
            raise FileNotFoundError(f"REF 模型不存在: {ref_model_path}")
        if not os.path.isfile(BATCH_INFER_BIN):
            raise FileNotFoundError(f"batch_inference 工具不存在: {BATCH_INFER_BIN}")

        # 初始化临时目录
        os.makedirs(_BATCH_DIR, exist_ok=True)
        os.makedirs(_OUTPUT_DIR, exist_ok=True)

        # 预热: 首次运行加载模型和初始化 RPU
        self._warmup()

    def _warmup(self):
        """预热: 用空数据跑一次推理，初始化 RPU 硬件"""
        dummy = np.zeros((1, 3, IMGSZ, IMGSZ), dtype=np.float32)
        inp_path = os.path.join(_BATCH_DIR, "batch_0_0.bin")
        dummy.tofile(inp_path)
        subprocess.run(
            [BATCH_INFER_BIN, self.ref_model, _BATCH_DIR, "--output", _OUTPUT_DIR],
            capture_output=True, text=True, timeout=10
        )

    def infer(self, img_bgr, conf_threshold=0.25, iou_threshold=0.45):
        """执行 YOLO 推理

        Args:
            img_bgr: BGR 图片 (numpy array)
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU 阈值

        Returns:
            detections: list of dict, 每个元素包含 cls, conf, x1, y1, x2, y2, cx, cy
        """
        t0 = time.time()

        # 1. 预处理
        img_nchw, lb_info = _letterbox(img_bgr, IMGSZ)
        t1 = time.time()

        # 2. 保存输入到 /dev/shm (内存文件系统, 避免磁盘 I/O)
        inp_path = os.path.join(_BATCH_DIR, "batch_0_0.bin")
        img_nchw.tofile(inp_path)
        t2 = time.time()

        # 3. 清理旧输出
        out_path = os.path.join(_OUTPUT_DIR, "batch_0_0.bin")
        if os.path.exists(out_path):
            os.remove(out_path)

        # 4. 运行 batch_inference
        result = subprocess.run(
            [BATCH_INFER_BIN, self.ref_model, _BATCH_DIR, "--output", _OUTPUT_DIR],
            capture_output=True, text=True, timeout=10
        )
        t3 = time.time()

        if not os.path.exists(out_path):
            print(f"[RhinoInfer] 推理失败: {result.stderr[-200:] if result.stderr else '无输出'}")
            return []

        # 5. 读取输出
        output = np.fromfile(out_path, dtype=np.float32).reshape(1, OUTPUT_DIM, NUM_ANCHORS)
        t4 = time.time()

        # 6. 后处理
        detections = postprocess(output, img_bgr.shape, lb_info,
                                 conf_threshold, iou_threshold)
        t5 = time.time()

        print(f"[RhinoInfer] 预处理:{(t1-t0)*1000:.0f}ms 保存:{(t2-t1)*1000:.0f}ms "
              f"推理:{(t3-t2)*1000:.0f}ms 读取:{(t4-t3)*1000:.0f}ms "
              f"后处理:{(t5-t4)*1000:.0f}ms 总计:{(t5-t0)*1000:.0f}ms "
              f"检测到{len(detections)}个目标")

        return detections

    def detect_ab(self, img_bgr, conf_threshold=0.25, strict_ab=True, min_boxes=1):
        """兼容 detect_ab 接口的检测函数

        Returns:
            (best_dict, annotated_img) 或 (None, annotated_img)
            best_dict: {"a": {...}, "b": {...}} 每个值包含 cls, conf, x1, y1, x2, y2, cx, cy
        """
        # 多阈值回退 (与 chassis_correct_all.py 中的逻辑一致)
        conf_thresholds = []
        for c in [conf_threshold, 0.15, 0.10]:
            if c >= 0.10 and c not in conf_thresholds:
                conf_thresholds.append(c)
        if 0.10 not in conf_thresholds:
            conf_thresholds.append(0.10)

        last_detections = []
        last_annotated = None

        for try_conf in conf_thresholds:
            detections = self.infer(img_bgr, conf_threshold=try_conf)
            last_detections = detections
            last_annotated = draw_annotated(img_bgr, detections)

            # 调试输出
            if detections:
                box_info = ", ".join([
                    f"{d['cls']}={d['conf']:.2f}@({d['cx']:.0f},{d['cy']:.0f})"
                    for d in detections
                ])
                print(f"  [Rhino检测 conf={try_conf}] {len(detections)}框: {box_info}")
            else:
                print(f"  [Rhino检测 conf={try_conf}] 0框")

            # 构建 best dict
            best = {}
            for det in detections:
                cls = det["cls"]
                if cls not in best or det["conf"] > best[cls]["conf"]:
                    best[cls] = det

            if "a" in best and "b" in best:
                return best, last_annotated

        # strict_ab=True 位置回退: 2个高置信度框水平排列时, 按x坐标左=a右=b
        # 几何约束放宽: y_diff < max(avg_h * 2.0, 60) 防止小框被误拒
        if strict_ab and len(last_detections) >= 2:
            sorted_by_conf = sorted(last_detections, key=lambda b: b["conf"], reverse=True)
            top2 = sorted_by_conf[:2]
            if top2[0]["conf"] >= 0.25 and top2[1]["conf"] >= 0.25:
                x0, y0 = top2[0]["cx"], top2[0]["cy"]
                x1, y1 = top2[1]["cx"], top2[1]["cy"]
                avg_h = ((top2[0]["y2"] - top2[0]["y1"]) + (top2[1]["y2"] - top2[1]["y1"])) / 2
                y_diff = abs(y0 - y1)
                x_dist = abs(x0 - x1)
                # 放宽: 相对阈值 (2倍框高) 或 绝对阈值 (60px) 取大者
                y_threshold = max(avg_h * 2.0, 60.0)
                if y_diff < y_threshold and x_dist > 30:
                    left = top2[0] if x0 < x1 else top2[1]
                    right = top2[1] if x0 < x1 else top2[0]
                    print(f"  [Rhino位置回退] 2个高置信度框水平排列 "
                          f"(conf={top2[0]['conf']:.2f}/{top2[1]['conf']:.2f}, "
                          f"y_diff={y_diff:.0f}<{y_threshold:.0f}, x_dist={x_dist:.0f}>30), 左=a 右=b")
                    return ({"a": dict(left, cls="a"), "b": dict(right, cls="b")},
                            last_annotated)

        # 回退: topN / single (仅 strict_ab=False 时)
        if not strict_ab and len(last_detections) >= min_boxes:
            sorted_boxes = sorted(last_detections, key=lambda b: b["conf"], reverse=True)
            if len(sorted_boxes) >= 2:
                return {"a": sorted_boxes[0], "b": sorted_boxes[1]}, last_annotated
            elif min_boxes == 1:
                return {"a": sorted_boxes[0], "b": sorted_boxes[0]}, last_annotated

        return None, last_annotated


# ── 自测 ──
if __name__ == "__main__":
    import sys

    print("=== 辉羲 RPU 推理模块自测 ===")
    engine = RhinoInfer(DEFAULT_REF_MODEL)
    print(f"模型: {DEFAULT_REF_MODEL}")
    print(f"预热完成\n")

    # 测试: 生成测试图片
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = engine.infer(test_img, conf_threshold=0.25)
    print(f"\n测试图片检测结果: {len(detections)} 个目标")

    # 测试 detect_ab 接口
    best, annotated = engine.detect_ab(test_img, conf_threshold=0.25)
    if best:
        print(f"detect_ab 结果: {list(best.keys())}")
    else:
        print("detect_ab 结果: None (无检测)")

    print("\n=== 自测完成 ===")
