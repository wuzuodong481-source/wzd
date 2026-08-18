#!/usr/bin/env python3
"""
放货三步纠偏总控程序 (辉羲 RPU 芯片推理)

使用 jitai_new.ref 模型, 通过辉羲 RPU 芯片加速推理。
按顺序执行三步纠偏: 前后 → 角度 → 左右

与 chassis_correct_all.py 的区别:
  - 推理模型: jitai_new.ref (放货场景标记检测)
  - 推理方式: 辉羲 RPU 芯片 (无网络延迟, ~100ms/次)

用法:
  python jitai_correct_all.py
  python jitai_correct_all.py --target-depth 750
  python jitai_correct_all.py --dry-run          # 只检测不移动
  python jitai_correct_all.py --skip-fb           # 跳过前后纠偏
  python jitai_correct_all.py --skip-yaw          # 跳过角度纠偏
  python jitai_correct_all.py --skip-lr           # 跳过左右纠偏
"""
import argparse
import os
import sys
import time
import glob
import json
import base64
import threading

import cv2
import numpy as np

# ===================== 辉羲 RPU 推理 =====================
USE_RHINO_INFER = True
RHINO_REF_MODEL = "/data/wzd/jitai_new.ref"
if USE_RHINO_INFER:
    try:
        from rhino_infer import RhinoInfer
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rhino_infer", "/data/wzd/rhino_infer.py")
        rhino_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rhino_mod)
        RhinoInfer = rhino_mod.RhinoInfer

# ===================== 复用 chassis_correct_all 的相机/底盘接口 =====================
import chassis_correct_all as cca
# 复用: 相机, 深度, 底盘, 绘图, 辅助
capture_color = cca.capture_color
capture_color_and_depth = cca.capture_color_and_depth
get_depth_at_point = cca.get_depth_at_point
setup_minth = cca.setup_minth
move_chassis = cca.move_chassis
draw_top_bar = cca.draw_top_bar
draw_geometry = cca.draw_geometry
# 复用常量
GDK_SERVICES_DIR = cca.GDK_SERVICES_DIR
MINTH_DIR = cca.MINTH_DIR
IMAGE_SAVE_DIR = cca.IMAGE_SAVE_DIR
MQTT_BROKER = cca.MQTT_BROKER
MQTT_PORT = cca.MQTT_PORT
YOLO_IMGSZ = cca.YOLO_IMGSZ
YOLO_CONF = cca.YOLO_CONF
YOLO_MIN_CONF = cca.YOLO_MIN_CONF
COLOR_LINE_AB = cca.COLOR_LINE_AB
COLOR_MID = cca.COLOR_MID
WARMUP_WAIT = cca.WARMUP_WAIT

# ===================== 路径配置 =====================
MODEL_PATH = "/data/wzd/jitai.pt"
OUTPUT_DIR = "/data/wzd/correction_jitai"

# ===================== 纠偏参数 =====================
# 前后纠偏
DEFAULT_TARGET_DEPTH = 800      # 目标深度 mm
FB_THRESHOLD = 5                # 收敛阈值 mm
FB_MAX_ITER = 5                 # 最大迭代
FB_SETTLE_TIME = 2.0            # 移动后稳定时间
FB_MAX_SINGLE_MOVE = 0.30       # 单次最大移动 m
FB_MIN_DEPTH = 400              # 最小安全深度 mm (低于此值立即停止, 防止撞架)
FB_PRE_MOVE = True              # 小距离移动时预移动 (克服底盘静摩擦)
FB_PRE_MOVE_M = 0.05            # 预移动反向距离 (m)
FB_PRE_MOVE_THRESHOLD = 0.02    # 触发预移动的距离阈值 (m, 小于此值才预热, 仅微调时触发)
FB_GAIN = 1.0                   # 前后移动增益 (自适应初始值)
FB_MAX_DEPTH_DIFF = 300         # a/b 最大允许深度差 mm (超过则判定误检)

# 角度纠偏
YAW_THRESHOLD_DEG = 0.5         # 收敛阈值 度
YAW_MAX_ITER = 5                # 最大迭代
YAW_GAIN = 2.0                  # 初始增益 (自适应)
YAW_SETTLE_TIME = 1.5           # 移动后稳定时间
YAW_MAX_SINGLE_ROTATION = 15.0  # 单次最大旋转 度 (增大以突破静摩擦)
YAW_PRE_ROTATE = True           # 小角度旋转时预旋转 (克服底盘静摩擦)
YAW_PRE_ROTATE_DEG = 15.0       # 预旋转角度 (度)
YAW_PRE_ROTATE_THRESHOLD = 3.0  # 触发预旋转的角度阈值 (度, 小于此值才预热)

# 左右纠偏
LR_TARGET_X = 320               # 图像中心 x
LR_THRESHOLD = 5                # 收敛阈值 px
LR_MAX_ITER = 5                 # 最大迭代
LR_SETTLE_TIME = 1.5            # 移动后稳定时间
PX_TO_METER = 0.002584          # m/px (标定系数, 2026-08-13 重新标定, R²=0.98)
LR_PRE_MOVE = True              # 小距离移动时预横移 (克服底盘静摩擦)
LR_PRE_MOVE_M = 0.05            # 预横移反向距离 (m)
LR_PRE_MOVE_THRESHOLD = 0.01    # 触发预横移的距离阈值 (m, 小于此值才预热, 仅微调时触发)
LR_GAIN = 1.0                   # 左右移动增益

# 通用
WARMUP_WAIT = 0.2               # 预热拍照后等待时间
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# YOLO 推理
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_MIN_CONF = 0.10     # 最低有效置信度

# ===================== detect_ab (使用 RhinoInfer) =====================
def detect_ab(model, img_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, strict_ab=True, min_boxes=1):
    """YOLO 检测 a/b, 通过辉羲 RPU 芯片推理

    Returns: (best_dict, annotated) 或 (None, annotated)
    """
    if USE_RHINO_INFER and isinstance(model, RhinoInfer):
        return model.detect_ab(img_bgr, conf_threshold=conf,
                               strict_ab=strict_ab, min_boxes=min_boxes)

    # 本地回退 (不应到达此处)
    from ultralytics import YOLO
    if not isinstance(model, YOLO):
        model = YOLO(MODEL_PATH)
    results = model(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r0 = results[0]
    annotated = r0.plot()
    boxes = r0.boxes
    names = model.names

    all_boxes = []
    best = {}
    for box in boxes:
        cls_id = int(box.cls[0])
        box_conf = float(box.conf[0])
        cls_name = names.get(cls_id, str(cls_id))
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        info = {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                "cx": float((x1 + x2) / 2), "cy": float((y1 + y2) / 2),
                "conf": box_conf, "cls": cls_name}
        all_boxes.append(info)
        if cls_name not in best or box_conf > best[cls_name]["conf"]:
            best[cls_name] = info

    if all_boxes:
        box_info = ", ".join([f"{b['cls']}={b['conf']:.2f}@({b['cx']:.0f},{b['cy']:.0f})" for b in all_boxes])
        print(f"  [检测 conf={conf}] {len(all_boxes)}框: {box_info}")
    else:
        print(f"  [检测 conf={conf}] 0框")

    if "a" in best and "b" in best:
        return best, annotated

    # 位置回退
    if strict_ab and len(all_boxes) >= 2:
        sorted_by_conf = sorted(all_boxes, key=lambda b: b["conf"], reverse=True)
        top2 = sorted_by_conf[:2]
        if top2[0]["conf"] >= 0.25 and top2[1]["conf"] >= 0.25:
            x0, y0 = top2[0]["cx"], top2[0]["cy"]
            x1, y1 = top2[1]["cx"], top2[1]["cy"]
            avg_h = ((top2[0]["y2"] - top2[0]["y1"]) + (top2[1]["y2"] - top2[1]["y1"])) / 2
            if abs(y0 - y1) < avg_h * 0.8 and abs(x0 - x1) > 30:
                left = top2[0] if x0 < x1 else top2[1]
                right = top2[1] if x0 < x1 else top2[0]
                return {"a": dict(left, cls="a"), "b": dict(right, cls="b")}, annotated

    if not strict_ab and len(all_boxes) >= min_boxes:
        sorted_boxes = sorted(all_boxes, key=lambda b: b["conf"], reverse=True)
        if len(sorted_boxes) >= 2:
            return {"a": sorted_boxes[0], "b": sorted_boxes[1]}, annotated
        elif min_boxes == 1:
            return {"a": sorted_boxes[0], "b": sorted_boxes[0]}, annotated

    return None, annotated


# ===================== 前后纠偏 =====================
def step_fb_correct(model, g2, target_depth, output_dir, dry_run):
    """前后纠偏: 通过深度图对齐目标距离"""
    print(f"\n{'='*60}")
    print(f"[步骤 1/3] 前后纠偏 (目标深度: {target_depth}mm)")
    print(f"{'='*60}")

    curr_gain = FB_GAIN
    prev_delta = None
    prev_move = None

    for iteration in range(1, FB_MAX_ITER + 1):
        print(f"\n--- 前后纠偏 迭代 {iteration}/{FB_MAX_ITER} ---")

        # 拍照
        print(f"[FB-{iteration}] 拍照中...")
        color_img, depth_img = capture_color_and_depth()
        if color_img is None or depth_img is None:
            print(f"[FB-{iteration}] ✗ 拍照失败")
            return {"success": False, "step": "fb", "reason": "拍照失败", "color_img": None}

        # 检测
        best, annotated = detect_ab(model, color_img, strict_ab=False)
        if best is None:
            print(f"[FB-{iteration}] ✗ 未检测到 a/b")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  STATUS: NO_AB_DETECTED"], status="no_ab")
            return {"success": False, "step": "fb", "reason": "未检测到 a/b"}

        a_d = get_depth_at_point(depth_img, int(best["a"]["cx"]), int(best["a"]["cy"]))
        b_d = get_depth_at_point(depth_img, int(best["b"]["cx"]), int(best["b"]["cy"]))
        avg_d = (a_d + b_d) / 2
        delta = avg_d - target_depth

        print(f"[FB-{iteration}] a={a_d:.0f}mm b={b_d:.0f}mm avg={avg_d:.0f}mm delta={delta:+.0f}mm")

        # 安全检查
        if avg_d < FB_MIN_DEPTH:
            print(f"[FB-{iteration}] ⚠⚠ 安全终止: 深度 {avg_d:.0f}mm < {FB_MIN_DEPTH}mm")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  SAFETY STOP",
                              f"avg={avg_d:.0f}mm < {FB_MIN_DEPTH}mm"], status="safety")
            return {"success": False, "step": "fb", "reason": f"安全终止: 深度过近 {avg_d:.0f}mm"}

        # 保存标注图
        ts = time.strftime("%H%M%S")
        iter_img = annotated.copy()
        cv2.circle(iter_img, (int(best["a"]["cx"]), int(best["a"]["cy"])), 4, (0,0,255), -1)
        cv2.circle(iter_img, (int(best["b"]["cx"]), int(best["b"]["cy"])), 4, (0,0,255), -1)
        cv2.putText(iter_img, f"a:{a_d:.0f}mm", (int(best["a"]["cx"])+8, int(best["a"]["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        cv2.putText(iter_img, f"b:{b_d:.0f}mm", (int(best["b"]["cx"])+8, int(best["b"]["cy"])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        iter_img = draw_top_bar(iter_img, [f"FB iter:{iteration} a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm delta:{delta:+.0f}mm"])
        cv2.imwrite(os.path.join(output_dir, f"01_fb_iter{iteration}_{ts}.jpg"), iter_img)

        # 收敛判断
        if abs(delta) < FB_THRESHOLD:
            print(f"[FB-{iteration}] ✓ 收敛 (|{delta:.0f}| < {FB_THRESHOLD})")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  CONVERGED",
                              f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm delta:{delta:+.0f}mm"])
            return {"success": True, "step": "fb", "converged": True,
                    "a_depth": a_d, "b_depth": b_d, "avg_depth": avg_d,
                    "color_img": color_img}

        # 自适应增益
        if prev_delta is not None and prev_move is not None and abs(delta) > 0.1:
            actual_change = prev_delta - delta
            ratio = actual_change / (prev_move * 1000) if abs(prev_move * 1000) > 0.1 else 1.0
            if abs(ratio) > 0.01:
                ideal_gain = 1.0 / ratio
                new_gain = 0.5 * curr_gain + 0.5 * ideal_gain
                new_gain = max(0.3, min(2.0, new_gain))
                print(f"[FB-{iteration}] 自适应: ratio={ratio:.2f} gain {curr_gain:.2f}→{new_gain:.2f}")
                curr_gain = new_gain

        # 计算移动量
        move_m = delta / 1000.0 * curr_gain
        if abs(move_m) > FB_MAX_SINGLE_MOVE:
            move_m = np.sign(move_m) * FB_MAX_SINGLE_MOVE

        print(f"[FB-{iteration}] 移动: {move_m*1000:+.0f}mm ({'前进' if move_m > 0 else '后退'}) [gain={curr_gain:.2f}]")

        if dry_run:
            print(f"[FB-{iteration}] (DRY RUN)")
            prev_delta = delta
            prev_move = move_m
            continue

        # 预移动 (小距离时克服静摩擦)
        if FB_PRE_MOVE and abs(move_m) < FB_PRE_MOVE_THRESHOLD:
            pre_m = FB_PRE_MOVE_M * (-1 if move_m > 0 else 1)
            print(f"[FB-{iteration}] 小距离预热: 先后退{FB_PRE_MOVE_M*1000:.0f}mm")
            move_chassis(g2, dx_m=pre_m)
            time.sleep(FB_SETTLE_TIME)
            move_m = move_m - pre_m

        ok = move_chassis(g2, dx_m=move_m)
        if not ok:
            return {"success": False, "step": "fb", "reason": "底盘移动失败"}

        time.sleep(FB_SETTLE_TIME)
        prev_delta = delta
        prev_move = move_m

    # 最大迭代
    print(f"[FB-{iteration}] 达到最大迭代")
    _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                     [f"FB iter:{iteration}  MAX_ITER",
                      f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm delta:{delta:+.0f}mm"], status="max_iter")
    return {"success": True, "step": "fb", "converged": False,
            "a_depth": a_d, "b_depth": b_d, "avg_depth": avg_d,
            "color_img": color_img}


# ===================== 角度纠偏 =====================
def step_yaw_correct(model, g2, output_dir, dry_run, reuse_img=None):
    """角度纠偏: 通过 a/b 连线斜率对齐底盘 Yaw"""
    print(f"\n{'='*60}")
    print(f"[步骤 2/3] 角度纠偏 (目标: angle → 0°)")
    print(f"{'='*60}")

    curr_gain = YAW_GAIN
    prev_angle = None
    prev_yaw_deg = None

    for iteration in range(1, YAW_MAX_ITER + 1):
        print(f"\n--- 角度纠偏 迭代 {iteration}/{YAW_MAX_ITER} ---")

        # 拍照
        if iteration == 1 and reuse_img is not None:
            print(f"[YAW-{iteration}] 复用上一步彩色图")
            img = reuse_img
        else:
            print(f"[YAW-{iteration}] 拍照中...")
            img = capture_color()

        # 检测 (可重试)
        best = None
        annotated = None
        new_photo_count = 0
        while new_photo_count < 3:
            if new_photo_count > 0 or (iteration == 1 and reuse_img is None) or iteration > 1:
                if new_photo_count > 0:
                    print(f"[YAW-{iteration}] 拍照中... (第{new_photo_count+1}次)")
                    img = capture_color()
                    time.sleep(0.3)
            best, annotated = detect_ab(model, img, strict_ab=True)
            if best is not None:
                break
            print(f"[YAW-{iteration}] ✗ 未同时检测到 a/b, 重试...")
            new_photo_count += 1

        if best is None:
            print(f"[YAW-{iteration}] ✗ 多次拍照仍未检测到 a/b")
            _save_step_final(output_dir, "02_yaw", iteration, img, annotated,
                             [f"YAW iter:{iteration}  STATUS: NO_AB_DETECTED"], status="no_ab")
            return {"success": False, "step": "yaw", "reason": "未检测到 a/b"}

        a_cx, a_cy = best["a"]["cx"], best["a"]["cy"]
        b_cx, b_cy = best["b"]["cx"], best["b"]["cy"]
        dx = b_cx - a_cx
        slope = (b_cy - a_cy) / dx if abs(dx) > 1e-6 else float("inf")
        angle = np.degrees(np.arctan(slope)) if abs(slope) < 1e6 else 90.0

        mid_x = (a_cx + b_cx) / 2
        mid_y = (a_cy + b_cy) / 2

        print(f"[YAW-{iteration}] slope={slope:+.4f} angle={angle:+.2f}° gain={curr_gain:.2f}")

        # 保存标注图
        ts = time.strftime("%H%M%S")
        iter_img = draw_geometry(annotated.copy(), best, mid_x, mid_y, iteration,
                                 [f"YAW iter:{iteration} slope:{slope:+.4f} angle:{angle:+.2f}deg gain:{curr_gain:.2f}"])
        cv2.imwrite(os.path.join(output_dir, f"02_yaw_iter{iteration}_{ts}.jpg"), iter_img)

        # 收敛判断
        if abs(angle) < YAW_THRESHOLD_DEG:
            print(f"[YAW-{iteration}] ✓ 收敛 (|{angle:.2f}| < {YAW_THRESHOLD_DEG})")
            _save_step_final(output_dir, "02_yaw", iteration, img, annotated,
                             [f"YAW iter:{iteration}  CONVERGED",
                              f"slope:{slope:+.4f}  angle:{angle:+.2f}deg  gain:{curr_gain:.2f}"])
            return {"success": True, "step": "yaw", "converged": True,
                    "final_angle": angle, "color_img": img}

        # 自适应增益
        if prev_angle is not None and prev_yaw_deg is not None:
            actual_change = prev_angle - angle
            expected_change = -prev_yaw_deg
            if abs(expected_change) > 0.1:
                ratio = actual_change / expected_change
                if abs(ratio) > 0.01:
                    ideal_gain = 1.0 / ratio
                    new_gain = 0.5 * curr_gain + 0.5 * ideal_gain
                    new_gain = max(0.5, min(3.0, new_gain))
                    print(f"[YAW-{iteration}] 自适应: ratio={ratio:.2f} gain {curr_gain:.2f}→{new_gain:.2f}")
                    curr_gain = new_gain

        # 计算旋转量
        delta_yaw_rad = -np.arctan(slope) * curr_gain
        delta_yaw_deg = np.degrees(delta_yaw_rad)
        if abs(delta_yaw_deg) > YAW_MAX_SINGLE_ROTATION:
            delta_yaw_deg = np.sign(delta_yaw_deg) * YAW_MAX_SINGLE_ROTATION
            delta_yaw_rad = np.radians(delta_yaw_deg)

        print(f"[YAW-{iteration}] 旋转: {delta_yaw_deg:+.2f}°")

        if dry_run:
            print(f"[YAW-{iteration}] (DRY RUN)")
            prev_angle = angle
            prev_yaw_deg = delta_yaw_deg
            continue

        # 预旋转
        if YAW_PRE_ROTATE and abs(delta_yaw_deg) < YAW_PRE_ROTATE_THRESHOLD:
            pre_deg = YAW_PRE_ROTATE_DEG * (-1 if delta_yaw_deg > 0 else 1)
            print(f"[YAW-{iteration}] [预旋转] 先转 {pre_deg:+.1f}° 再补偿 (克服静摩擦)")
            move_chassis(g2, yaw_rad=np.radians(pre_deg))
            time.sleep(YAW_SETTLE_TIME)
            delta_yaw_rad = delta_yaw_rad - np.radians(pre_deg)

        ok = move_chassis(g2, yaw_rad=delta_yaw_rad)
        if not ok:
            return {"success": False, "step": "yaw", "reason": "底盘旋转失败"}

        time.sleep(YAW_SETTLE_TIME)
        prev_angle = angle
        prev_yaw_deg = delta_yaw_deg

    # 最大迭代
    print(f"[YAW-{iteration}] 达到最大迭代")
    _save_step_final(output_dir, "02_yaw", iteration, img, annotated,
                     [f"YAW iter:{iteration}  MAX_ITER",
                      f"slope:{slope:+.4f}  angle:{angle:+.2f}deg  gain:{curr_gain:.2f}"], status="max_iter")
    return {"success": True, "step": "yaw", "converged": False,
            "final_angle": angle, "color_img": img}


# ===================== 左右纠偏 =====================
def step_lr_correct(model, g2, output_dir, dry_run, reuse_img=None):
    """左右纠偏: 通过 a/b 中点 x 对齐图像中心"""
    print(f"\n{'='*60}")
    print(f"[步骤 3/3] 左右纠偏 (目标: mid_x → {LR_TARGET_X}px)")
    print(f"{'='*60}")

    curr_gain = LR_GAIN
    prev_delta_px = None
    prev_move_m = None

    for iteration in range(1, LR_MAX_ITER + 1):
        print(f"\n--- 左右纠偏 迭代 {iteration}/{LR_MAX_ITER} ---")

        # 拍照
        if iteration == 1 and reuse_img is not None:
            print(f"[LR-{iteration}] 复用上一步彩色图")
            img = reuse_img
        else:
            print(f"[LR-{iteration}] 拍照中...")
            img = capture_color()

        # 检测
        best, annotated = detect_ab(model, img, strict_ab=False, min_boxes=2)
        if best is None:
            print(f"[LR-{iteration}] ✗ 未检测到 a/b")
            _save_step_final(output_dir, "03_lr", iteration, img, annotated,
                             [f"LR iter:{iteration}  STATUS: NO_AB_DETECTED"], status="no_ab")
            return {"success": False, "step": "lr", "reason": "未检测到 a/b"}

        mid_x = (best["a"]["cx"] + best["b"]["cx"]) / 2
        mid_y = (best["a"]["cy"] + best["b"]["cy"]) / 2
        delta_px = mid_x - LR_TARGET_X

        print(f"[LR-{iteration}] mid_x={mid_x:.1f} delta={delta_px:+.1f}px")

        # 保存标注图
        ts = time.strftime("%H%M%S")
        iter_img = draw_geometry(annotated.copy(), best, mid_x, mid_y, iteration,
                                 [f"LR iter:{iteration} mid_x:{mid_x:.1f} delta:{delta_px:+.1f}px gain:{curr_gain:.2f}"])
        cv2.imwrite(os.path.join(output_dir, f"03_lr_iter{iteration}_{ts}.jpg"), iter_img)

        # 收敛判断
        if abs(delta_px) < LR_THRESHOLD:
            print(f"[LR-{iteration}] ✓ 收敛 (|{delta_px:.1f}| < {LR_THRESHOLD})")
            _save_step_final(output_dir, "03_lr", iteration, img, annotated,
                             [f"LR iter:{iteration}  CONVERGED",
                              f"mid_x:{mid_x:.1f}  delta:{delta_px:+.1f}px  gain:{curr_gain:.2f}"])
            return {"success": True, "step": "lr", "converged": True,
                    "final_delta_px": delta_px, "color_img": img}

        # 自适应增益
        if prev_delta_px is not None and prev_move_m is not None and abs(delta_px) > 0.1:
            actual_change = prev_delta_px - delta_px
            expected_change = -prev_move_m / PX_TO_METER
            if abs(expected_change) > 0.1:
                ratio = actual_change / expected_change
                if abs(ratio) > 0.01:
                    ideal_gain = 1.0 / ratio
                    new_gain = 0.5 * curr_gain + 0.5 * ideal_gain
                    new_gain = max(0.3, min(2.0, new_gain))
                    print(f"[LR-{iteration}] 自适应: ratio={ratio:.2f} gain {curr_gain:.2f}→{new_gain:.2f}")
                    curr_gain = new_gain

        # 计算移动量
        move_m = -delta_px * PX_TO_METER * curr_gain
        print(f"[LR-{iteration}] 移动: {move_m*1000:+.1f}mm ({'左' if move_m > 0 else '右'}) [gain={curr_gain:.2f}]")

        if dry_run:
            print(f"[LR-{iteration}] (DRY RUN)")
            prev_delta_px = delta_px
            prev_move_m = move_m
            continue

        # 预横移
        if LR_PRE_MOVE and abs(move_m) < LR_PRE_MOVE_THRESHOLD:
            pre_m = LR_PRE_MOVE_M * (-1 if move_m > 0 else 1)
            print(f"[LR-{iteration}] 小距离预热: 先{'右' if move_m > 0 else '左'}移{LR_PRE_MOVE_M*1000:.0f}mm")
            move_chassis(g2, dy_m=pre_m)
            time.sleep(LR_SETTLE_TIME)
            move_m = move_m - pre_m

        ok = move_chassis(g2, dy_m=move_m)
        if not ok:
            return {"success": False, "step": "lr", "reason": "底盘横移失败"}

        time.sleep(LR_SETTLE_TIME)
        prev_delta_px = delta_px
        prev_move_m = move_m

    # 最大迭代
    print(f"[LR-{iteration}] 达到最大迭代")
    _save_step_final(output_dir, "03_lr", iteration, img, annotated,
                     [f"LR iter:{iteration}  MAX_ITER",
                      f"mid_x:{mid_x:.1f}  delta:{delta_px:+.1f}px  gain:{curr_gain:.2f}"], status="max_iter")
    return {"success": True, "step": "lr", "converged": False,
            "final_delta_px": delta_px, "color_img": img}


# ===================== 辅助函数 =====================
def _save_step_final(output_dir, prefix, iteration, img, annotated, info_lines, status=""):
    """保存步骤最终结果图"""
    ts = time.strftime("%H%M%S")
    suffix = f"_{status}" if status else ""
    final_img = draw_top_bar(annotated.copy(), info_lines)
    path = os.path.join(output_dir, f"{prefix}_final_iter{iteration}_{ts}{suffix}.jpg")
    cv2.imwrite(path, final_img)
    print(f"[保存] 结果图已保存: {path}")


# ===================== 主函数 =====================
def parse_args():
    p = argparse.ArgumentParser(description="放货三步纠偏 (辉羲 RPU 芯片推理, jitai_new.ref)")
    p.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH, help="目标深度 mm")
    p.add_argument("--dry-run", action="store_true", help="只检测不移动")
    p.add_argument("--skip-fb", action="store_true", help="跳过前后纠偏")
    p.add_argument("--skip-yaw", action="store_true", help="跳过角度纠偏")
    p.add_argument("--skip-lr", action="store_true", help="跳过左右纠偏")
    return p.parse_args()


def main(args):
    t_total_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"{'='*60}")
    print(f"放货三步纠偏总控程序 (辉羲 RPU 芯片推理)")
    print(f"  模型: {RHINO_REF_MODEL}")
    print(f"  顺序: 前后 → 角度 → 左右")
    print(f"  目标深度: {args.target_depth}mm")
    print(f"  Dry Run: {args.dry_run}")
    steps = []
    if not args.skip_fb: steps.append("前后")
    if not args.skip_yaw: steps.append("角度")
    if not args.skip_lr: steps.append("左右")
    print(f"  执行步骤: {' → '.join(steps)}")
    print(f"{'='*60}")

    # 1. 加载模型
    if USE_RHINO_INFER:
        print(f"\n[初始化] 辉羲 RPU 芯片推理模式: {RHINO_REF_MODEL}")
        model = RhinoInfer(RHINO_REF_MODEL)
        print(f"[初始化] ✓ 辉羲 RPU 已就绪")
    else:
        from ultralytics import YOLO
        print(f"\n[初始化] 加载 YOLO 模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

    # 2. 连接机器人
    if args.dry_run:
        print(f"[初始化] DRY RUN 模式, 跳过机器人连接")
        g2 = None
    else:
        print(f"[初始化] 连接机器人...")
        g2 = setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")

    results = {}
    abort = False

    # 步骤 1: 前后纠偏
    if not args.skip_fb:
        r_fb = step_fb_correct(model, g2, args.target_depth, OUTPUT_DIR, args.dry_run)
        results["fb"] = r_fb
        if not r_fb["success"]:
            print(f"\n⚠⚠ 前后纠偏失败: {r_fb.get('reason')}")
            print(f"⚠⚠ 终止后续步骤, 停止机器人运动!")
            abort = True
        reuse_img = r_fb.get("color_img")
    else:
        print(f"\n[跳过] 前后纠偏")
        reuse_img = None

    # 步骤 2: 角度纠偏
    if not args.skip_yaw and not abort:
        r_yaw = step_yaw_correct(model, g2, OUTPUT_DIR, args.dry_run, reuse_img=reuse_img)
        results["yaw"] = r_yaw
        if not r_yaw["success"]:
            print(f"\n⚠⚠ 角度纠偏失败: {r_yaw.get('reason')}")
            print(f"⚠⚠ 终止后续步骤, 停止机器人运动!")
            abort = True
        reuse_img = r_yaw.get("color_img")
    elif not args.skip_yaw and abort:
        print(f"\n[跳过] 角度纠偏 (前后纠偏已异常终止)")
    else:
        print(f"\n[跳过] 角度纠偏")

    # 步骤 3: 左右纠偏
    if not args.skip_lr and not abort:
        r_lr = step_lr_correct(model, g2, OUTPUT_DIR, args.dry_run, reuse_img=reuse_img)
        results["lr"] = r_lr
        if not r_lr["success"]:
            print(f"\n⚠⚠ 左右纠偏失败: {r_lr.get('reason')}")
            print(f"⚠⚠ 终止后续步骤, 停止机器人运动!")
            abort = True
    elif not args.skip_lr and abort:
        print(f"\n[跳过] 左右纠偏 (前序纠偏已异常终止)")
    else:
        print(f"\n[跳过] 左右纠偏")

    # 异常终止
    if abort:
        print(f"\n{'='*60}")
        print(f"⚠⚠ 检测到异常, 已停止所有机器人运动")
        print(f"{'='*60}")

    # 释放连接
    if g2 is not None:
        g2.close()

    # 总结
    elapsed = time.time() - t_total_start
    print(f"\n{'='*60}")
    print(f"放货纠偏总结 (耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")
    for step_name, key in [("前后", "fb"), ("角度", "yaw"), ("左右", "lr")]:
        if key in results:
            r = results[key]
            if r.get("success") and r.get("converged"):
                status = "✓ 收敛"
                if key == "fb":
                    status += f"  a={r.get('a_depth',0):.0f}mm b={r.get('b_depth',0):.0f}mm avg={r.get('avg_depth',0):.0f}mm"
                elif key == "yaw":
                    status += f"  angle={r.get('final_angle',0):+.2f}°"
                elif key == "lr":
                    status += f"  delta={r.get('final_delta_px',0):+.1f}px"
            elif r.get("success") and not r.get("converged"):
                status = "⚠ 未完全收敛"
                if key == "fb":
                    status += f"  avg={r.get('avg_depth',0):.0f}mm"
                elif key == "yaw":
                    status += f"  angle={r.get('final_angle',0):+.2f}°"
                elif key == "lr":
                    status += f"  delta={r.get('final_delta_px',0):+.1f}px"
            else:
                status = f"✗ 失败 ({r.get('reason', '?')})"
            print(f"  {step_name}: {status}")
    print(f"{'='*60}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
