#!/usr/bin/env python3
"""
底盘 Yaw 角度纠偏程序

目标: 让 a/b 连线斜率归零 (angle < 阈值), 通过底盘 Yaw 旋转实现闭环纠偏。

原理:
  1. 拍照 → YOLO 检测 a/b → 计算连线斜率 slope
  2. 角度偏差 angle_deg = arctan(slope) * 180 / π
  3. 若 |angle_deg| > 阈值, 底盘 Yaw 旋转: delta_yaw = -arctan(slope) * YAW_GAIN
     (slope > 0 → 连线右下倾 → 底盘顺时针补偿 → yaw 为负)
  4. 等待稳定后重新拍照验证, 直到收敛或达到最大迭代次数

与腰部纠偏的区别:
  - 底盘 Yaw 纠偏: 修正底盘朝向偏差, 让相机位置和朝向同时归正
  - 腰部 Yaw 纠偏: 修正腰部朝向偏差, 仅让相机朝向归正 (位置仍偏)

为什么要做底盘 Yaw 纠偏:
  - 底盘是机器人基准, 底盘歪会导致所有传感器歪
  - 仅靠腰部"代偿"会让腰部越来越歪, 最终超出行程范围
  - 底盘 Yaw 修正后, 腰部基本不用动, 系统更稳定

检测要求:
  - 严格要求同时检测到 a 和 b (不使用回退策略)
  - 因为斜率/角度计算需要真正的 a/b 连线, 类别不能混淆

用法:
  python chassis_yaw_correct.py
  python chassis_yaw_correct.py --max-iter 5
  python chassis_yaw_correct.py --threshold-deg 0.5
  python chassis_yaw_correct.py --dry-run    # 只检测不移动
  python chassis_yaw_correct.py --yaw-gain 2.0  # 设置初始增益 (默认2.0, 自适应开启)
  python chassis_yaw_correct.py --no-adaptive-gain  # 关闭自适应增益, 使用固定增益
  python chassis_yaw_correct.py --invert     # 反转 yaw 方向 (若发现越纠越偏)
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
OUTPUT_DIR = "/data/wzd/correction_yaw"
# GDK 服务目录 (用于导入 agibot_gdk)
GDK_SERVICES_DIR = "/data/wxf/wxf0721/services"
# Minth 控制库路径 (用于调用机器人底盘旋转)
MINTH_DIR = "/data/wxf/wxf0721/runtime"

# ===================== 纠偏参数 =====================
# 收敛阈值 (角度, 度), |angle_deg| < 阈值 视为收敛
DEFAULT_THRESHOLD_DEG = 1.0
# 最大迭代次数
DEFAULT_MAX_ITER = 3
# Yaw 增益 (1.0 = 理论 1:1, 若实际欠补偿可增大, 过补偿可减小)
DEFAULT_YAW_GAIN = 2.0
# 每次移动后等待稳定时间 (秒)
SETTLE_TIME = 2.0
# 最大单次旋转角度 (度), 防止过大旋转
MAX_SINGLE_ROTATION_DEG = 10.0
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
#  YOLO 推理: 严格获取 a/b 中心点 (不使用回退策略)
# ═══════════════════════════════════════════════════════════

def get_ab_points(model, img_bgr, imgsz=640, conf=0.25):
    """对图像执行 YOLO 推理, 严格要求同时检测到 a 和 b

    返回: (mid_x, mid_y, slope, best_dict, annotated) 或 None

    策略: 先以 conf 阈值推理, 若 a 或 b 缺失, 自动降低置信度阈值重试 (0.25→0.15→0.1)
          这是为了应对底盘旋转后目标角度变化导致置信度暂时下降的情况。
    注: 底盘/腰部 Yaw 纠偏需要 a+b 来计算斜率, 但若旋转后某一类置信度下降,
        适当降阈值是合理的, 因为只要同时有 a 和 b 的框(即使置信度偏低), 连线斜率仍然有效。

    支持 REMOTE_INFER 模式: 通过 MQTT 请求远程 GPU 推理服务
    """
    # 自动降阈值重试: conf → 0.15 → 0.1 → 0.05 → 0.01 (去重)
    conf_thresholds = [conf, 0.15, 0.1, 0.05, 0.01]
    seen = set()
    conf_thresholds = [c for c in conf_thresholds if not (c in seen or seen.add(c))]

    for try_conf in conf_thresholds:
        # ===== 远程 GPU 推理 (MQTT) =====
        if REMOTE_INFER:
            resp = _remote_infer(img_bgr, imgsz=imgsz, conf=try_conf)
            if resp.get("success"):
                all_boxes = resp.get("all_boxes", [])
                best = resp.get("best", {})
                names = resp.get("names", {})
                annotated = _draw_annotated(img_bgr, all_boxes, names)
                if "a" in best and "b" in best:
                    if try_conf < conf:
                        print(f"[检测] 阈值降至 {try_conf:.2f} 后成功检测到 a/b "
                              f"(a.conf={best['a']['conf']:.2f}, b.conf={best['b']['conf']:.2f})")
                    a_cx, a_cy = best["a"]["cx"], best["a"]["cy"]
                    b_cx, b_cy = best["b"]["cx"], best["b"]["cy"]
                    mid_x = (a_cx + b_cx) / 2
                    mid_y = (a_cy + b_cy) / 2
                    dx = b_cx - a_cx
                    slope = (b_cy - a_cy) / dx if abs(dx) > 1e-6 else float("inf")
                    return mid_x, mid_y, slope, best, annotated
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

        best = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            box_conf = float(box.conf[0])
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in best or box_conf > best[cls_name]["conf"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                best[cls_name] = {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "cx": float((x1 + x2) / 2),
                    "cy": float((y1 + y2) / 2),
                    "conf": box_conf,
                }

        if "a" in best and "b" in best:
            if try_conf < conf:
                print(f"[检测] 阈值降至 {try_conf:.2f} 后成功检测到 a/b "
                      f"(a.conf={best['a']['conf']:.2f}, b.conf={best['b']['conf']:.2f})")
            a_cx, a_cy = best["a"]["cx"], best["a"]["cy"]
            b_cx, b_cy = best["b"]["cx"], best["b"]["cy"]
            mid_x = (a_cx + b_cx) / 2
            mid_y = (a_cy + b_cy) / 2
            dx = b_cx - a_cx
            slope = (b_cy - a_cy) / dx if abs(dx) > 1e-6 else float("inf")
            return mid_x, mid_y, slope, best, annotated

    return None


# ═══════════════════════════════════════════════════════════
#  在图像上绘制几何元素
# ═══════════════════════════════════════════════════════════

def draw_geometry(annotated, best, mid_x, mid_y, iteration, angle_deg, delta_yaw_rad):
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
    length = max(np.sqrt(dx * dx + dy * dy), 1e-6)
    perp_x = -dy / length
    perp_y = dx / length
    L = max(h, w)
    cv2.line(img,
             (int(mid_x - perp_x * L), int(mid_y - perp_y * L)),
             (int(mid_x + perp_x * L), int(mid_y + perp_y * L)),
             COLOR_PERP_BISECTOR, 1)

    # 4. 图像中心竖直线
    cv2.line(img, (320, 0), (320, h), COLOR_IMG_CENTER, 1)

    # 5. 顶部数据栏
    slope = (b_cy - a_cy) / (b_cx - a_cx) if abs(b_cx - a_cx) > 1e-6 else float("inf")
    slope_str = f"{slope:.4f}" if abs(slope) < 1e6 else "inf"
    delta_yaw_deg = np.degrees(delta_yaw_rad)

    line1 = f"iter:{iteration}  slope:{slope_str}  angle:{angle_deg:+.2f}deg"
    line2 = f"mid:({mid_x:.1f},{mid_y:.1f})  yaw_move:{delta_yaw_deg:+.2f}deg"
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

def run_correction(model, g2, threshold_deg, max_iter, output_dir, imgsz, conf,
                   yaw_gain, invert, dry_run, adaptive_gain=True):
    """执行底盘 Yaw 纠偏闭环

    返回: dict 纠偏结果
    """
    total_yaw_rad = 0.0
    history = []  # [(iter, angle_deg, yaw_rad, gain), ...]
    # 自适应增益: 记录上一次的角度和旋转量, 用于计算实际补偿比例
    prev_angle_deg = None
    prev_delta_yaw_deg = None
    curr_gain = yaw_gain

    print(f"\n{'=' * 60}")
    print(f"开始底盘 Yaw 角度纠偏")
    print(f"  目标: angle → 0° (a/b 连线水平)")
    print(f"  阈值: |angle| < {threshold_deg}°")
    print(f"  最大迭代: {max_iter}")
    print(f"  Yaw 增益: {yaw_gain}" + (" (自适应)" if adaptive_gain else " (固定)"))
    print(f"  方向反转: {'是' if invert else '否'}")
    print(f"  单次最大旋转: {MAX_SINGLE_ROTATION_DEG}°")
    if dry_run:
        print(f"  *** DRY RUN: 只检测不移动 ***")
    print(f"{'=' * 60}")

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
            res = get_ab_points(model, img, imgsz, conf)
            if res is not None:
                break
            if attempt < 3:
                print(f"[{iteration}] 未同时检测到 a/b, 等待重试...")
                time.sleep(1.0)
        if res is None:
            print(f"[{iteration}] ✗ 3次拍照均未同时检测到 a/b, 终止纠偏")
            cv2.imwrite(os.path.join(output_dir, f"yaw_fail_iter{iteration}.jpg"), img)
            return {"success": False, "reason": "未同时检测到 a/b", "iteration": iteration, "history": history}

        mid_x, mid_y, slope, best, annotated = res

        # 2. 计算角度偏差
        angle_deg = np.degrees(np.arctan(slope)) if abs(slope) < 1e6 else 90.0
        print(f"[{iteration}] 中点=({mid_x:.1f}, {mid_y:.1f})  "
              f"a.conf={best['a']['conf']:.2f}  b.conf={best['b']['conf']:.2f}")
        print(f"[{iteration}] 斜率: slope={slope:+.4f}  角度: angle={angle_deg:+.2f}°")

        # 3. 判断收敛
        if abs(angle_deg) < threshold_deg:
            print(f"[{iteration}] ✓ 已收敛 (|{angle_deg:.2f}| < {threshold_deg})")
            final_img = draw_geometry(annotated, best, mid_x, mid_y, iteration, angle_deg, 0.0)
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(output_dir, f"yaw_final_iter{iteration}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            history.append((iteration, angle_deg, 0.0, curr_gain))
            return {
                "success": True,
                "converged": True,
                "iteration": iteration,
                "final_angle_deg": angle_deg,
                "final_slope": slope,
                "total_yaw_rad": total_yaw_rad,
                "history": history,
                "final_image": final_path,
            }

        # 3.5 自适应增益: 根据上一次的实际补偿比例调整增益
        if adaptive_gain and prev_angle_deg is not None and prev_delta_yaw_deg is not None:
            # 实际角度变化 (prev → curr)
            actual_change = prev_angle_deg - angle_deg
            # 理论预期角度变化 = -prev_delta_yaw_deg (旋转 yaw 度, 角度应反向变化 yaw 度)
            expected_change = -prev_delta_yaw_deg
            if abs(expected_change) > 0.1:  # 避免除零
                ratio = actual_change / expected_change  # 补偿比例, <1 表示欠补偿
                if ratio > 0.1:  # 合理范围才更新
                    ideal_gain = curr_gain / ratio
                    # 平滑更新 (50% 旧值 + 50% 新值), 避免剧烈波动
                    new_gain = 0.5 * curr_gain + 0.5 * ideal_gain
                    # 限制范围 [0.5, 5.0]
                    new_gain = max(0.5, min(5.0, new_gain))
                    print(f"[{iteration}] 自适应增益: ratio={ratio:.2f} "
                          f"gain {curr_gain:.2f} → {new_gain:.2f}")
                    curr_gain = new_gain
                else:
                    print(f"[{iteration}] 自适应增益: ratio={ratio:.2f} 异常, 保持 gain={curr_gain:.2f}")

        # 4. 计算底盘 Yaw 旋转量
        # slope > 0 (b 在 a 下方, 图像坐标 y 向下) → 连线右下倾 → 底盘需顺时针补偿 → yaw 为负
        # 默认: delta_yaw = -arctan(slope) * gain
        # invert: 反转方向 (若发现越纠越偏, 加 --invert)
        sign = -1.0 if not invert else 1.0
        delta_yaw_rad = sign * np.arctan(slope) * curr_gain

        # 限制单次最大旋转
        delta_yaw_deg = np.degrees(delta_yaw_rad)
        if abs(delta_yaw_deg) > MAX_SINGLE_ROTATION_DEG:
            delta_yaw_deg = np.sign(delta_yaw_deg) * MAX_SINGLE_ROTATION_DEG
            delta_yaw_rad = np.radians(delta_yaw_deg)
            print(f"[{iteration}] ⚠ 旋转量过大, 限制为 {delta_yaw_deg:+.2f}°")

        print(f"[{iteration}] 需旋转: yaw = {delta_yaw_deg:+.2f}° ({'逆时针' if delta_yaw_rad > 0 else '顺时针'})")

        # 5. 绘制并保存当前状态图 (旋转前)
        state_img = draw_geometry(annotated, best, mid_x, mid_y, iteration, angle_deg, delta_yaw_rad)
        ts = time.strftime("%Y%m%d_%H%M%S")
        state_path = os.path.join(output_dir, f"yaw_iter{iteration}_before_{ts}.jpg")
        cv2.imwrite(state_path, state_img)

        # 6. 执行旋转
        if dry_run:
            print(f"[{iteration}] (DRY RUN 跳过旋转)")
        else:
            ok = move_chassis_relative(g2, yaw_rad=delta_yaw_rad)
            if not ok:
                print(f"[{iteration}] ✗ 旋转失败, 终止纠偏")
                history.append((iteration, angle_deg, delta_yaw_rad, curr_gain))
                return {"success": False, "reason": "底盘旋转失败", "iteration": iteration, "history": history}
            total_yaw_rad += delta_yaw_rad
            time.sleep(SETTLE_TIME)

        # 更新自适应增益跟踪变量
        prev_angle_deg = angle_deg
        prev_delta_yaw_deg = delta_yaw_deg
        history.append((iteration, angle_deg, delta_yaw_rad, curr_gain))

    # 达到最大迭代仍未收敛
    print(f"\n⚠ 达到最大迭代 {max_iter}, 未完全收敛")
    # 最后拍一张验证
    if not dry_run:
        print(f"[最终] 拍照验证...")
        _ = capture_head_color()  # 预热刷新缓冲区
        time.sleep(1.0)
        img = capture_head_color()
        res = get_ab_points(model, img, imgsz, conf)
        if res is not None:
            mid_x, mid_y, slope, best, annotated = res
            angle_deg = np.degrees(np.arctan(slope)) if abs(slope) < 1e6 else 90.0
            final_img = draw_geometry(annotated, best, mid_x, mid_y, max_iter, angle_deg, 0.0)
            ts = time.strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(output_dir, f"yaw_final_iter{max_iter}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            print(f"[最终] angle={angle_deg:+.2f}°  slope={slope:+.4f}")
            return {
                "success": True,
                "converged": False,
                "iteration": max_iter,
                "final_angle_deg": angle_deg,
                "final_slope": slope,
                "total_yaw_rad": total_yaw_rad,
                "history": history,
                "final_image": final_path,
            }

    return {
        "success": False,
        "converged": False,
        "iteration": max_iter,
        "total_yaw_rad": total_yaw_rad,
        "history": history,
    }


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    global SETTLE_TIME, REMOTE_INFER, REMOTE_INFER_FALLBACK
    parser = argparse.ArgumentParser(description="底盘 Yaw 角度纠偏: 让 a/b 连线斜率归零")
    parser.add_argument("--model", default=MODEL_PATH, help=f"YOLO 模型路径 (默认: {MODEL_PATH})")
    parser.add_argument("--output", default=OUTPUT_DIR, help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--threshold-deg", type=float, default=DEFAULT_THRESHOLD_DEG,
                        help=f"收敛阈值(角度,度) (默认: {DEFAULT_THRESHOLD_DEG})")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help=f"最大迭代次数 (默认: {DEFAULT_MAX_ITER})")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--yaw-gain", type=float, default=DEFAULT_YAW_GAIN,
                        help=f"Yaw 增益 (1.0=理论1:1, 欠补偿增大, 过补偿减小) (默认: {DEFAULT_YAW_GAIN})")
    parser.add_argument("--invert", action="store_true", help="反转 Yaw 方向 (若发现越纠越偏时使用)")
    parser.add_argument("--no-adaptive-gain", action="store_true",
                        help="关闭自适应增益 (默认开启, 程序根据实际补偿比例自动调整增益)")
    parser.add_argument("--settle", type=float, default=SETTLE_TIME, help=f"旋转后稳定时间(秒) (默认: {SETTLE_TIME})")
    parser.add_argument("--broker", default=MQTT_BROKER, help=f"MQTT broker (默认: {MQTT_BROKER})")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help=f"MQTT 端口 (默认: {MQTT_PORT})")
    parser.add_argument("--dry-run", action="store_true", help="只检测不旋转 (测试用)")
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
        model, g2, args.threshold_deg, args.max_iter,
        args.output, args.imgsz, args.conf,
        args.yaw_gain, args.invert, args.dry_run,
        adaptive_gain=not args.no_adaptive_gain
    )

    # 释放连接
    if g2 is not None:
        g2.close()

    # 打印总结
    print(f"\n{'=' * 60}")
    print(f"纠偏总结")
    print(f"{'=' * 60}")
    print(f"成功:     {result.get('success', False)}")
    print(f"收敛:     {result.get('converged', False)}")
    print(f"迭代次数: {result.get('iteration', 0)}")
    if 'final_angle_deg' in result:
        print(f"最终角度: {result['final_angle_deg']:+.2f}°")
        print(f"最终斜率: {result['final_slope']:+.4f}")
    if 'total_yaw_rad' in result:
        print(f"总旋转量: {np.degrees(result['total_yaw_rad']):+.2f}°")
    if 'history' in result and result['history']:
        print(f"\n迭代历史:")
        print(f"  {'iter':<6}{'angle_deg':<14}{'yaw_deg':<12}{'gain':<8}")
        for it, ang, yaw, gain in result['history']:
            print(f"  {it:<6}{ang:<+14.2f}{np.degrees(yaw):<+12.2f}{gain:<8.2f}")
    if 'final_image' in result:
        print(f"\n最终图像: {result['final_image']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
