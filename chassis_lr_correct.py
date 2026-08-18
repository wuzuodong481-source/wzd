#!/usr/bin/env python3
"""
底盘左右纠偏程序

目标: 让 a/b 连线中点 x 对齐图像中心 (320px), 通过底盘 Y 方向横移实现闭环纠偏。

原理:
  1. 拍照 → YOLO 检测 a/b → 计算连线中点 mid_x
  2. 像素偏差 delta_px = mid_x - 320
  3. 若 |delta_px| > 阈值, 底盘横移: delta_y = -delta_px * PX_TO_METER
     (mid_x > 320 → 目标偏右 → 底盘左移 → y 为正)
  4. 等待稳定后重新拍照验证, 直到收敛或达到最大迭代次数

转换系数来自 chassis_lr_calibrate.py 的标定结果 (R²=0.9832):
  PX_TO_METER = 0.001964 m/px  (1 像素 ≈ 1.96 毫米)

用法:
  python chassis_lr_correct.py
  python chassis_lr_correct.py --max-iter 5
  python chassis_lr_correct.py --threshold 3
  python chassis_lr_correct.py --dry-run    # 只检测不移动
"""
import argparse
import os
import sys
import time
import json
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== 路径配置 (可后续修改) =====================
# 模型路径
MODEL_PATH = "/data/wzd/best_new.pt"
# 输出目录 (保存纠偏过程图像)
OUTPUT_DIR = "/data/wzd/correction"
# GDK 服务目录 (用于导入 agibot_gdk)
GDK_SERVICES_DIR = "/data/wxf/wxf0721/services"
# Minth 控制库路径 (用于调用机器人底盘横移)
MINTH_DIR = "/data/wxf/wxf0721/runtime"

# ===================== 纠偏参数 =====================
# 标定得到的像素→米转换系数 (来自 chassis_lr_calibrate.py)
PX_TO_METER = 0.001964  # m/px
# 图像中心 x (640 宽度的中心)
TARGET_X = 320
# 收敛阈值 (像素), |delta_px| < 阈值 视为收敛
DEFAULT_THRESHOLD = 5
# 最大迭代次数
DEFAULT_MAX_ITER = 3
# 每次移动后等待稳定时间 (秒)
SETTLE_TIME = 1.5
# MQTT 配置 (仅用于 minth 底盘控制)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# ===================== 远程 GPU 推理配置 =====================
REMOTE_INFER = False                      # 是否启用远程推理 (默认本地)
REMOTE_INFER_TOPIC_REQ = "/yolo/infer/request"
REMOTE_INFER_TOPIC_RSP = "/yolo/infer/response"
REMOTE_INFER_TIMEOUT = 5.0                # 单次推理超时 s
REMOTE_INFER_FALLBACK = True              # 远程失败时回退到本地 CPU 推理

# ===================== 几何绘制参数 =====================
COLOR_LINE_AB = (0, 255, 0)       # a/b 连线: 绿色
COLOR_MID = (0, 0, 255)           # 中点: 红色
COLOR_PERP_BISECTOR = (255, 0, 0) # 中垂线: 蓝色
COLOR_IMG_CENTER = (255, 255, 0)  # 图像中心竖直线: 青色
COLOR_BAR_BG = (0, 0, 0)          # 顶部数据栏背景: 黑色
COLOR_BAR_TEXT = (255, 255, 255)  # 顶部数据栏文字: 白色

# GDK 相机对象 (延迟初始化)
_gdk_camera = None


# ═══════════════════════════════════════════════════════════
#  通过 GDK 直接拍照头部彩色相机 (不依赖 MQTT 相机流)
# ═══════════════════════════════════════════════════════════

def _init_gdk():
    """初始化 GDK 相机接口"""
    global _gdk_camera
    if _gdk_camera is not None:
        return _gdk_camera
    if GDK_SERVICES_DIR not in sys.path:
        sys.path.insert(0, GDK_SERVICES_DIR)
    try:
        import agibot_gdk
        _gdk_camera = agibot_gdk.Camera()
        print(f"[GDK] 相机接口初始化成功")
        return _gdk_camera
    except Exception as e:
        print(f"[GDK] 初始化失败: {e}, 将回退到 MQTT 方式")
        return None


def capture_head_color(broker=MQTT_BROKER, port=MQTT_PORT, timeout=10.0):
    """拍摄头部彩色相机图像 (优先 GDK, 回退 MQTT)"""
    cam = _init_gdk()
    if cam is not None:
        try:
            import agibot_gdk
            img = cam.get_latest_image(agibot_gdk.CameraType.kHeadColor, timeout * 1000.0)
            if img is not None and img.data is not None:
                encoding = getattr(img, 'encoding', None)
                color_format = getattr(img, 'color_format', None)
                raw = img.data
                if encoding == agibot_gdk.Encoding.JPEG:
                    nparr = np.frombuffer(raw, dtype=np.uint8)
                    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if bgr is not None:
                        return bgr
                elif color_format in (agibot_gdk.ColorFormat.RGB, agibot_gdk.ColorFormat.BGR, agibot_gdk.ColorFormat.GRAY8):
                    nparr = np.frombuffer(raw, dtype=np.uint8)
                    if color_format == agibot_gdk.ColorFormat.RGB:
                        rgb = nparr.reshape((img.height, img.width, 3))
                        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    elif color_format == agibot_gdk.ColorFormat.BGR:
                        bgr = nparr.reshape((img.height, img.width, 3))
                    else:
                        gray = nparr.reshape((img.height, img.width))
                        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                    return bgr
        except Exception as e:
            print(f"[GDK] 拍照失败, 回退 MQTT: {e}")

    # 回退: MQTT 订阅相机流
    import base64 as _b64
    import json as _json
    import paho.mqtt.client as mqtt

    received = {"img": None}
    topic_ctrl = "/humanoid/camera/control"
    topic_data = "/humanoid/camera/data"

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(topic_data, qos=0)
            client.publish(topic_ctrl, _json.dumps({"command": "start"}), qos=0)

    def on_message(client, userdata, msg):
        try:
            payload = _json.loads(msg.payload.decode("utf-8"))
            b64 = payload.get("head_color")
            if b64:
                buf = _b64.b64decode(b64)
                nparr = np.frombuffer(buf, dtype=np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    received["img"] = bgr
                    client.disconnect()
        except Exception:
            pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)

    t_start = time.time()
    while received["img"] is None and time.time() - t_start < timeout:
        client.loop(timeout=0.2)
    try:
        client.publish(topic_ctrl, _json.dumps({"command": "stop"}), qos=0)
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass

    if received["img"] is None:
        raise RuntimeError(f"在 {timeout}s 内未收到相机数据")
    return received["img"]


# ═══════════════════════════════════════════════════════════
#  远程 GPU 推理客户端 (MQTT)
# ═══════════════════════════════════════════════════════════

def _remote_infer(img_bgr, imgsz=640, conf=0.25,
                  broker=MQTT_BROKER, port=MQTT_PORT,
                  timeout=REMOTE_INFER_TIMEOUT):
    """通过 MQTT 请求远程 GPU 推理服务

    Returns: dict {success, all_boxes, best, names, elapsed_ms} 或 {success:False, error}
    """
    import paho.mqtt.client as mqtt

    req_id = f"{int(time.time()*1000)}-{os.getpid()}"
    result = {"success": False, "error": "timeout"}
    done_event = threading.Event()

    def on_msg(client, userdata, msg):
        try:
            resp = json.loads(msg.payload.decode("utf-8"))
            if resp.get("request_id") == req_id:
                result.clear()
                result.update(resp)
                done_event.set()
        except Exception:
            pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_msg
    try:
        client.connect(broker, port, keepalive=10)
        client.loop_start()
        client.subscribe(REMOTE_INFER_TOPIC_RSP, qos=0)

        # 编码图片 (JPEG → base64)
        ok, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return {"success": False, "error": "图片编码失败"}
        img_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        req = {
            "request_id": req_id,
            "image": img_b64,
            "imgsz": int(imgsz),
            "conf": float(conf),
        }
        client.publish(REMOTE_INFER_TOPIC_REQ, json.dumps(req), qos=0)

        if not done_event.wait(timeout=timeout):
            return {"success": False, "error": f"超时 {timeout}s"}
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        try:
            client.loop_stop()
            client.disconnect()
        except Exception:
            pass


def _draw_annotated(img_bgr, all_boxes, names=None):
    """根据检测框本地绘制标注图 (替代原 r0.plot())"""
    annotated = img_bgr.copy()
    color_map = {"a": (0, 255, 0), "b": (0, 165, 255)}
    for box in all_boxes:
        cls = box.get("cls", "?")
        color = color_map.get(cls, (255, 0, 0))
        x1, y1 = int(box["x1"]), int(box["y1"])
        x2, y2 = int(box["x2"]), int(box["y2"])
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {box.get('conf', 0):.2f}"
        cv2.putText(annotated, label, (x1, max(y1 - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return annotated


# ═══════════════════════════════════════════════════════════
#  YOLO 推理: 获取 a/b 中心点
# ═══════════════════════════════════════════════════════════

def get_midpoint(model, img_bgr, imgsz=640, conf=0.25):
    """对图像执行 YOLO 推理, 返回 (mid_x, mid_y, best_dict, annotated) 或 None
    策略:
      1. 先以 conf 阈值推理, 若 a 或 b 缺失, 自动降低置信度阈值重试 (0.25→0.15→0.1→0.05→0.01)
      2. 若始终只有 a 或只有 b, 回退到取置信度最高的 2 个框作为中点 (用于左右纠偏)

    支持 REMOTE_INFER 模式: 通过 MQTT 请求远程 GPU 推理服务
    """
    conf_thresholds = [conf, 0.15, 0.1, 0.05, 0.01]
    seen = set()
    conf_thresholds = [c for c in conf_thresholds if not (c in seen or seen.add(c))]

    last_annotated = None
    last_all_boxes = None

    for try_conf in conf_thresholds:
        # ===== 远程 GPU 推理 (MQTT) =====
        if REMOTE_INFER:
            resp = _remote_infer(img_bgr, imgsz=imgsz, conf=try_conf)
            if resp.get("success"):
                all_boxes = resp.get("all_boxes", [])
                best = resp.get("best", {})
                names = resp.get("names", {})
                annotated = _draw_annotated(img_bgr, all_boxes, names)
                last_annotated = annotated
                last_all_boxes = all_boxes
                if "a" in best and "b" in best:
                    if try_conf < conf:
                        print(f"    [!] 使用置信度阈值 {try_conf} 检测到 a/b")
                    mid_x = (best["a"]["cx"] + best["b"]["cx"]) / 2
                    mid_y = (best["a"]["cy"] + best["b"]["cy"]) / 2
                    return mid_x, mid_y, best, annotated
                continue
            else:
                # 远程失败
                err = resp.get("error", "未知")
                print(f"  [远程推理] 失败: {err}")
                if REMOTE_INFER_FALLBACK and model is not None:
                    print(f"  [远程推理] 回退到本地 CPU 推理")
                else:
                    return None

        # ===== 本地推理 (默认, 或远程失败回退) =====
        results = model(img_bgr, imgsz=imgsz, conf=try_conf, verbose=False)
        r0 = results[0]
        annotated = r0.plot()
        boxes = r0.boxes
        names = model.names

        # 收集所有框 (按置信度排序)
        all_boxes = []
        best = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            box_conf = float(box.conf[0])
            cls_name = names.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            info = {
                "x1": float(x1), "y1": float(y1),
                "x2": float(x2), "y2": float(y2),
                "cx": float((x1 + x2) / 2),
                "cy": float((y1 + y2) / 2),
                "conf": box_conf,
                "cls": cls_name,
            }
            all_boxes.append(info)
            if cls_name not in best or box_conf > best[cls_name]["conf"]:
                best[cls_name] = info

        last_annotated = annotated
        last_all_boxes = all_boxes

        if "a" in best and "b" in best:
            if try_conf < conf:
                print(f"    [!] 使用置信度阈值 {try_conf} 检测到 a/b")
            mid_x = (best["a"]["cx"] + best["b"]["cx"]) / 2
            mid_y = (best["a"]["cy"] + best["b"]["cy"]) / 2
            return mid_x, mid_y, best, annotated

    # 回退: a/b 无法同时检测到, 取置信度最高的 2 个框
    if last_all_boxes and len(last_all_boxes) >= 2:
        sorted_boxes = sorted(last_all_boxes, key=lambda b: b["conf"], reverse=True)
        top2 = sorted_boxes[:2]
        print(f"    [!] a/b 无法同时检测, 回退到 top2 置信度框: "
              f"{top2[0]['cls']}({top2[0]['conf']:.2f}) + {top2[1]['cls']}({top2[1]['conf']:.2f})")
        best = {"a": top2[0], "b": top2[1]}
        mid_x = (top2[0]["cx"] + top2[1]["cx"]) / 2
        mid_y = (top2[0]["cy"] + top2[1]["cy"]) / 2
        return mid_x, mid_y, best, last_annotated

    return None


# ═══════════════════════════════════════════════════════════
#  在图像上绘制几何元素 (与 draw_line_slope.py 一致)
# ═══════════════════════════════════════════════════════════

def draw_geometry(annotated, best, mid_x, mid_y, target_x, iteration, delta_px, delta_y_m):
    """在标注图上绘制连线、中点、中垂线、图像中心竖直线、顶部数据栏"""
    img = annotated.copy()
    h, w = img.shape[:2]

    a_cx, a_cy = best["a"]["cx"], best["a"]["cy"]
    b_cx, b_cy = best["b"]["cx"], best["b"]["cy"]

    # 1. a/b 连线
    cv2.line(img, (int(a_cx), int(a_cy)), (int(b_cx), int(b_cy)), COLOR_LINE_AB, 2)

    # 2. 中点
    cv2.circle(img, (int(mid_x), int(mid_y)), 6, COLOR_MID, -1)

    # 3. 中垂线 (过中点, 垂直于 a/b 连线)
    dx = b_cx - a_cx
    dy = b_cy - a_cy
    length = max(np.sqrt(dx*dx + dy*dy), 1e-6)
    # 垂直方向 (旋转 90°)
    perp_x = -dy / length
    perp_y = dx / length
    L = max(h, w)
    cv2.line(img,
             (int(mid_x - perp_x * L), int(mid_y - perp_y * L)),
             (int(mid_x + perp_x * L), int(mid_y + perp_y * L)),
             COLOR_PERP_BISECTOR, 1)

    # 4. 图像中心竖直线
    cv2.line(img, (target_x, 0), (target_x, h), COLOR_IMG_CENTER, 1)

    # 5. 顶部数据栏
    slope = (b_cy - a_cy) / (b_cx - a_cx) if abs(b_cx - a_cx) > 1e-6 else float("inf")
    if abs(slope) < 1e-6:
        slope_str = "inf"
        angle_deg = 90.0
    else:
        slope_str = f"{slope:.4f}"
        angle_deg = np.degrees(np.arctan(1.0 / slope)) if abs(slope) > 1e-6 else 0.0

    line1 = f"iter:{iteration}  slope:{slope_str}  angle:{angle_deg:.1f}deg"
    line2 = f"mid:({mid_x:.1f},{mid_y:.1f})  target:{target_x}  delta:{delta_px:+.1f}px  move:{delta_y_m*1000:+.1f}mm"
    bar_h = 55
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    font_scale = 0.6
    thickness = 2
    cv2.putText(bar, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BAR_TEXT, thickness)
    cv2.putText(bar, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BAR_TEXT, thickness)
    img = np.vstack([bar, img])

    return img


# ═══════════════════════════════════════════════════════════
#  机器人控制 (通过 minth)
# ═══════════════════════════════════════════════════════════

def setup_minth():
    """加载 minth 模块并初始化 G2 控制"""
    if MINTH_DIR not in sys.path:
        sys.path.insert(0, MINTH_DIR)
    import minth
    g2 = minth.G2(broker=MQTT_BROKER, port=MQTT_PORT, timeout=60)
    return g2


def move_chassis_relative(g2, dx_m=0.0, dy_m=0.0, yaw_rad=0.0):
    """底盘相对运动 (dx 前, dy 左, yaw 逆时针)"""
    return g2._send_and_wait("go_rel", {"x": dx_m, "y": dy_m, "yaw_rad": yaw_rad})


# ═══════════════════════════════════════════════════════════
#  纠偏主流程
# ═══════════════════════════════════════════════════════════

def run_correction(model, g2, target_x, threshold, max_iter, output_dir, imgsz, conf, dry_run):
    """执行左右纠偏闭环

    返回: dict 纠偏结果
    """
    total_move_m = 0.0
    history = []  # [(iter, mid_x, delta_px, move_m), ...]

    print(f"\n{'='*60}")
    print(f"开始左右纠偏")
    print(f"  目标 x: {target_x} px (图像中心)")
    print(f"  阈值:   {threshold} px (约 {threshold * PX_TO_METER * 1000:.1f} mm)")
    print(f"  最大迭代: {max_iter}")
    print(f"  转换系数: {PX_TO_METER} m/px (1 像素 ≈ {PX_TO_METER*1000:.2f} mm)")
    if dry_run:
        print(f"  *** DRY RUN: 只检测不移动 ***")
    print(f"{'='*60}")

    for iteration in range(1, max_iter + 1):
        print(f"\n--- 迭代 {iteration}/{max_iter} ---")

        # 1. 拍照 (最多重试3次)
        # 先预热拍照刷新缓冲区 (GDK回退MQTT时取的是缓存帧, 需先刷新)
        if not dry_run:
            _ = capture_head_color()
            time.sleep(1.0)
        res = None
        for attempt in range(1, 4):
            print(f"[{iteration}] 拍照中... (第{attempt}次)")
            img = capture_head_color()
            res = get_midpoint(model, img, imgsz, conf)
            if res is not None:
                break
            if attempt < 3:
                print(f"[{iteration}] 未检测到 a/b, 等待重试...")
                time.sleep(1.0)
        if res is None:
            print(f"[{iteration}] ✗ 3次拍照均未检测到 a/b, 终止纠偏")
            cv2.imwrite(os.path.join(output_dir, f"correct_fail_iter{iteration}.jpg"), img)
            return {"success": False, "reason": "未检测到 a/b", "iteration": iteration, "history": history}

        mid_x, mid_y, best, annotated = res

        # 2. 计算偏差
        delta_px = mid_x - target_x
        print(f"[{iteration}] 中点=({mid_x:.1f}, {mid_y:.1f})  "
              f"a.conf={best['a']['conf']:.2f}  b.conf={best['b']['conf']:.2f}")
        print(f"[{iteration}] 偏差: delta_px = {delta_px:+.1f} px  (目标: {target_x})")

        # 3. 判断收敛
        if abs(delta_px) < threshold:
            print(f"[{iteration}] ✓ 已收敛 (|{delta_px:.1f}| < {threshold})")
            # 保存最终图
            final_img = draw_geometry(annotated, best, mid_x, mid_y, target_x, iteration, delta_px, 0.0)
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(output_dir, f"correct_final_iter{iteration}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            history.append((iteration, mid_x, delta_px, 0.0))
            return {
                "success": True,
                "converged": True,
                "iteration": iteration,
                "final_mid_x": mid_x,
                "final_delta_px": delta_px,
                "total_move_m": total_move_m,
                "history": history,
                "final_image": final_path,
            }

        # 4. 计算移动距离
        # delta_px > 0: 中点偏右 → 底盘需左移 → y 为正
        # delta_px < 0: 中点偏左 → 底盘需右移 → y 为负
        delta_y_m = -delta_px * PX_TO_METER
        print(f"[{iteration}] 需移动: y = {delta_y_m*1000:+.1f} mm ({'左' if delta_y_m > 0 else '右'})")

        # 5. 绘制并保存当前状态图 (移动前)
        state_img = draw_geometry(annotated, best, mid_x, mid_y, target_x, iteration, delta_px, delta_y_m)
        ts = time.strftime("%Y%m%d_%H%M%S")
        state_path = os.path.join(output_dir, f"correct_iter{iteration}_before_{ts}.jpg")
        cv2.imwrite(state_path, state_img)

        # 6. 执行移动
        if dry_run:
            print(f"[{iteration}] (DRY RUN 跳过移动)")
        else:
            ok = move_chassis_relative(g2, dy_m=delta_y_m)
            if not ok:
                print(f"[{iteration}] ✗ 移动失败, 终止纠偏")
                history.append((iteration, mid_x, delta_px, delta_y_m))
                return {"success": False, "reason": "底盘移动失败", "iteration": iteration, "history": history}
            total_move_m += delta_y_m
            time.sleep(SETTLE_TIME)

        history.append((iteration, mid_x, delta_px, delta_y_m))

    # 达到最大迭代仍未收敛
    print(f"\n⚠ 达到最大迭代 {max_iter}, 未完全收敛")
    # 最后拍一张验证 (最多重试3次)
    if not dry_run:
        print(f"[最终] 拍照验证...")
        _ = capture_head_color()  # 预热刷新缓冲区
        time.sleep(1.0)
        res = None
        for attempt in range(1, 4):
            print(f"[最终] 拍照中... (第{attempt}次)")
            img = capture_head_color()
            res = get_midpoint(model, img, imgsz, conf)
            if res is not None:
                break
            if attempt < 3:
                time.sleep(1.0)
        if res is not None:
            mid_x, mid_y, best, annotated = res
            delta_px = mid_x - target_x
            final_img = draw_geometry(annotated, best, mid_x, mid_y, target_x, max_iter, delta_px, 0.0)
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(output_dir, f"correct_final_iter{max_iter}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            print(f"[最终] 中点=({mid_x:.1f}, {mid_y:.1f})  偏差: {delta_px:+.1f} px")
            return {
                "success": True,
                "converged": False,
                "iteration": max_iter,
                "final_mid_x": mid_x,
                "final_delta_px": delta_px,
                "total_move_m": total_move_m,
                "history": history,
                "final_image": final_path,
            }

    return {
        "success": False,
        "converged": False,
        "iteration": max_iter,
        "total_move_m": total_move_m,
        "history": history,
    }


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    global SETTLE_TIME, REMOTE_INFER, REMOTE_INFER_FALLBACK
    parser = argparse.ArgumentParser(description="底盘左右纠偏: 对齐中点 x 到图像中心")
    parser.add_argument("--model", default=MODEL_PATH, help=f"YOLO 模型路径 (默认: {MODEL_PATH})")
    parser.add_argument("--output", default=OUTPUT_DIR, help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--target-x", type=int, default=TARGET_X, help=f"目标 x 像素 (默认: {TARGET_X})")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"收敛阈值(像素) (默认: {DEFAULT_THRESHOLD}, 约 {DEFAULT_THRESHOLD*PX_TO_METER*1000:.1f}mm)")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help=f"最大迭代次数 (默认: {DEFAULT_MAX_ITER})")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--settle", type=float, default=SETTLE_TIME, help=f"移动后稳定时间(秒) (默认: {SETTLE_TIME})")
    parser.add_argument("--broker", default=MQTT_BROKER, help=f"MQTT broker (默认: {MQTT_BROKER})")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help=f"MQTT 端口 (默认: {MQTT_PORT})")
    parser.add_argument("--dry-run", action="store_true", help="只检测不移动 (测试用)")
    parser.add_argument("--remote-infer", action="store_true",
                        help="启用远程 GPU 推理 (MQTT, 需先启动 yolo_infer_server.py)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="远程推理失败时不回退到本地 CPU 推理")
    args = parser.parse_args()

    # 应用远程推理配置
    if args.remote_infer:
        REMOTE_INFER = True
        print(f"[配置] 远程 GPU 推理: 启用")
    if args.no_fallback:
        REMOTE_INFER_FALLBACK = False

    SETTLE_TIME = args.settle

    os.makedirs(args.output, exist_ok=True)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    # 加载模型 (远程推理时可选加载本地模型作回退)
    if REMOTE_INFER:
        if REMOTE_INFER_FALLBACK:
            print(f"[1] 远程推理模式 + 本地回退, 加载本地模型: {args.model}")
            model = YOLO(args.model)
        else:
            print(f"[1] 远程推理模式 (无回退), 跳过本地模型加载")
            model = None
    else:
        print(f"[1] 加载 YOLO 模型: {args.model}")
        model = YOLO(args.model)
    print(f"    task={model.task if model else '(远程模式)'}, names={model.names if model else '(远程模式)'}")

    # 初始化机器人控制
    if args.dry_run:
        print(f"[2] DRY RUN 模式, 跳过机器人连接")
        g2 = None
    else:
        print(f"[2] 连接机器人 (MQTT {args.broker}:{args.port})...")
        g2 = setup_minth()
        print(f"    ✓ Minth 已就绪")

    # 执行纠偏
    result = run_correction(
        model, g2, args.target_x, args.threshold, args.max_iter,
        args.output, args.imgsz, args.conf, args.dry_run
    )

    # 释放连接
    if g2 is not None:
        g2.close()

    # 打印总结
    print(f"\n{'='*60}")
    print(f"纠偏总结")
    print(f"{'='*60}")
    print(f"成功:     {result.get('success', False)}")
    print(f"收敛:     {result.get('converged', False)}")
    print(f"迭代次数: {result.get('iteration', 0)}")
    if 'final_mid_x' in result:
        print(f"最终中点 x: {result['final_mid_x']:.1f} px")
        print(f"最终偏差:   {result['final_delta_px']:+.1f} px (约 {abs(result['final_delta_px'])*PX_TO_METER*1000:.1f} mm)")
    if 'total_move_m' in result:
        print(f"总移动量:   {result['total_move_m']*1000:+.1f} mm")
    if 'history' in result and result['history']:
        print(f"\n迭代历史:")
        print(f"  {'iter':<6}{'mid_x':<12}{'delta_px':<14}{'move_mm':<12}")
        for it, mx, dpx, dm in result['history']:
            print(f"  {it:<6}{mx:<12.1f}{dpx:<+14.1f}{dm*1000:<+12.1f}")
    if 'final_image' in result:
        print(f"\n最终图像: {result['final_image']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
