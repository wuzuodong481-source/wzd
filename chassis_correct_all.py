#!/usr/bin/env python3
"""
底盘综合纠偏总控程序

按顺序执行三步纠偏: 前后 → 角度 → 左右
  1. 前后纠偏: 通过深度图对齐目标距离 (默认 750mm)
  2. 角度纠偏: 通过 a/b 连线斜率对齐底盘 Yaw (自适应增益)
  3. 左右纠偏: 通过 a/b 中点 x 对齐图像中心 (320px)

优化点 (相比分别运行三个程序):
  - 模型只加载一次 (省 ~10s)
  - MQTT/Minth 只连接一次 (省 ~4s)
  - 前后纠偏拍的彩色+深度图, 角度纠偏可直接复用彩色图 (省 1 次拍照 ~3s)
  - 预热等待时间从 1.0s 缩短到 0.5s
  - SETTLE_TIME 从 2-3s 缩短到 1.5-2s

用法:
  python chassis_correct_all.py
  python chassis_correct_all.py --target-depth 750
  python chassis_correct_all.py --dry-run          # 只检测不移动
  python chassis_correct_all.py --skip-fb           # 跳过前后纠偏
  python chassis_correct_all.py --skip-yaw          # 跳过角度纠偏
  python chassis_correct_all.py --skip-lr           # 跳过左右纠偏
  python chassis_correct_all.py --max-iter 3        # 各步最大迭代次数
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
from ultralytics import YOLO

# ===================== 辉羲 RPU 推理 =====================
USE_RHINO_INFER = True               # 使用辉羲芯片加速推理 (True=芯片, False=YOLO CPU/远程)
RHINO_REF_MODEL = "/data/wzd/best_new.ref"
if USE_RHINO_INFER:
    try:
        from rhino_infer import RhinoInfer
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("rhino_infer", "/data/wzd/rhino_infer.py")
        rhino_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(rhino_mod)
        RhinoInfer = rhino_mod.RhinoInfer

# ===================== 路径配置 =====================
MODEL_PATH = "/data/wzd/best_new.pt"
OUTPUT_DIR = "/data/wzd/correction_all"
GDK_SERVICES_DIR = "/data/wxf/wxf0721/services"
MINTH_DIR = "/data/wxf/wxf0721/runtime"
IMAGE_SAVE_DIR = "/data/wxf/wxf0721/images"

# ===================== 纠偏参数 =====================
# 前后纠偏
DEFAULT_TARGET_DEPTH = 750      # 目标深度 mm
FB_THRESHOLD = 5                # 收敛阈值 mm
FB_MAX_ITER = 5                 # 最大迭代
FB_SETTLE_TIME = 2.0            # 移动后稳定时间
FB_MAX_SINGLE_MOVE = 0.30       # 单次最大移动 m
FB_MIN_DEPTH = 400              # 最小安全深度 mm (低于此值立即停止, 防止撞架)
FB_PRE_MOVE = True              # 小距离移动时预移动 (克服底盘静摩擦)
FB_PRE_MOVE_M = 0.05            # 预移动反向距离 (m)
FB_PRE_MOVE_THRESHOLD = 0.02    # 触发预移动的距离阈值 (m, 小于此值才预热, 仅微调时触发)
FB_GAIN = 1.0                   # 前后移动增益 (自适应初始值)

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
PX_TO_METER = 0.002584          # m/px (标定系数, 2026-08-13 重新标定, 剔除异常点, R²=0.98)
LR_PRE_MOVE = True              # 小距离移动时预横移 (克服底盘静摩擦)
LR_PRE_MOVE_M = 0.05            # 预横移反向距离 (m)
LR_PRE_MOVE_THRESHOLD = 0.01    # 触发预横移的距离阈值 (m, 小于此值才预热, 仅微调时触发)
LR_GAIN = 1.0                   # 左右移动增益 (标定系数已准确, 无需增益补偿)

# 通用
WARMUP_WAIT = 0.2               # 预热拍照后等待时间 (原 0.5s)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# YOLO 推理
YOLO_IMGSZ = 640
YOLO_CONF = 0.25
YOLO_MIN_CONF = 0.10     # 最低有效置信度, 低于此值的检测框视为误检 (防止围栏/背景误检)

# 远程 GPU 推理 (MQTT)
REMOTE_INFER = False                      # 是否启用远程推理 (默认本地)
REMOTE_INFER_TOPIC_REQ = "/yolo/infer/request"
REMOTE_INFER_TOPIC_RSP = "/yolo/infer/response"
REMOTE_INFER_TIMEOUT = 5.0                # 单次推理超时 s
REMOTE_INFER_FALLBACK = True              # 远程失败时回退到本地 CPU 推理

# ===================== 绘制参数 =====================
COLOR_LINE_AB = (0, 255, 0)
COLOR_MID = (0, 0, 255)
COLOR_PERP = (255, 0, 0)
COLOR_CENTER = (255, 255, 0)
COLOR_TEXT = (255, 255, 255)

# ===================== 全局状态 =====================
_gdk_camera = None


# ═══════════════════════════════════════════════════════════
#  相机: GDK 彩色图 + MQTT 深度图
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
        return _gdk_camera
    except Exception as e:
        return None


def capture_color(broker=MQTT_BROKER, port=MQTT_PORT, timeout=10.0):
    """拍摄头部彩色图 (优先 GDK, 回退 MQTT)"""
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
                        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    elif color_format == agibot_gdk.ColorFormat.BGR:
                        return nparr.reshape((img.height, img.width, 3))
                    else:
                        gray = nparr.reshape((img.height, img.width))
                        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception:
            pass

    # 回退: MQTT
    received = {"img": None}
    topic_ctrl = "/humanoid/camera/control"
    topic_data = "/humanoid/camera/data"

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe(topic_data, qos=0)
            client.publish(topic_ctrl, json.dumps({"command": "start"}), qos=0)

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            b64 = payload.get("head_color")
            if b64:
                buf = base64.b64decode(b64)
                nparr = np.frombuffer(buf, dtype=np.uint8)
                bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    received["img"] = bgr
                    client.disconnect()
        except Exception:
            pass

    import paho.mqtt.client as mqtt
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)
    t_start = time.time()
    while received["img"] is None and time.time() - t_start < timeout:
        client.loop(timeout=0.2)
    try:
        client.publish(topic_ctrl, json.dumps({"command": "stop"}), qos=0)
        client.disconnect()
    except Exception:
        pass
    if received["img"] is None:
        raise RuntimeError(f"在 {timeout}s 内未收到相机数据")
    return received["img"]


def _send_save_photo(broker=MQTT_BROKER, port=MQTT_PORT, timeout=15.0):
    """通过 MQTT 发送 save_photo 命令, 等待完成"""
    import paho.mqtt.client as mqtt
    done = {"ok": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe("/humanoid/camera/done", qos=0)
            client.publish("/humanoid/camera/control",
                           json.dumps({"command": "save_photo",
                                       "cameras": ["kHeadColor", "kHeadDepth"]}), qos=0)

    def on_message(client, userdata, msg):
        if msg.topic == "/humanoid/camera/done":
            done["ok"] = True
            client.disconnect()

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)
    t_start = time.time()
    while not done["ok"] and time.time() - t_start < timeout:
        client.loop(timeout=0.2)
    try:
        client.disconnect()
    except Exception:
        pass
    return done["ok"]


def _read_latest_color_and_depth(existing_color, existing_depth, tried_depth=None):
    """读取最新的彩色图和深度图文件

    Args:
        existing_color: 之前的 color 文件集合
        existing_depth: 之前的 depth 文件集合
        tried_depth: 已尝试过但异常的 depth 文件集合 (跳过这些)
    """
    if tried_depth is None:
        tried_depth = set()
    color_files = sorted(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadColor_*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadDepth_raw_*.raw")))

    new_color = [f for f in color_files if f not in existing_color]
    new_depth = [f for f in depth_files if f not in existing_depth and f not in tried_depth]

    color_bgr = None
    depth_2d = None

    if new_color:
        color_bgr = cv2.imread(new_color[-1])
    if new_depth:
        raw = np.fromfile(new_depth[-1], dtype=np.uint16)
        # 尺寸校验: 期望 400*640=256000 个 uint16
        if raw.size != 400 * 640:
            print(f"[WARN] 深度图尺寸异常: {raw.size} != {400*640}, 跳过: {new_depth[-1]}")
            depth_2d = None
        else:
            depth_2d = raw.reshape((400, 640))

    return color_bgr, depth_2d


def capture_color_and_depth(broker=MQTT_BROKER, port=MQTT_PORT, timeout=15.0):
    """单次拍照获取彩色+深度图 (优化: 轮询文件出现, 省去 done 等待和 WARMUP_WAIT)

    改进: 每3秒重发一次 save_photo 命令, 防止相机丢命令导致超时。
    同时当深度帧异常时自动重发命令请求新帧, 替代外部预拍照刷新。
    """
    existing_c = set(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadColor_*.jpg")))
    existing_d = set(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadDepth_raw_*.raw")))

    import paho.mqtt.client as mqtt
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    try:
        client.connect(broker, port, keepalive=10)
        client.loop_start()
        client.publish("/humanoid/camera/control",
                       json.dumps({"command": "save_photo",
                                   "cameras": ["kHeadColor", "kHeadDepth"]}), qos=0)
    except Exception:
        pass

    t_start = time.time()
    color_bgr, depth_2d = None, None
    tried_depth = set()  # 已尝试过的异常深度文件
    last_resend = t_start
    while time.time() - t_start < timeout:
        color_bgr, depth_2d = _read_latest_color_and_depth(existing_c, existing_d, tried_depth)
        if color_bgr is not None and depth_2d is not None:
            break
        # depth 异常: 标记最新文件以跳过, 等待新文件
        if depth_2d is None:
            depth_files = sorted(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadDepth_raw_*.raw")))
            new_d_now = [f for f in depth_files if f not in existing_d and f not in tried_depth]
            if new_d_now:
                tried_depth.add(new_d_now[-1])
        # 每3秒重发一次 save_photo (防止丢命令或请求新帧替代异常帧)
        now = time.time()
        if now - last_resend >= 3.0:
            try:
                client.publish("/humanoid/camera/control",
                               json.dumps({"command": "save_photo",
                                           "cameras": ["kHeadColor", "kHeadDepth"]}), qos=0)
            except Exception:
                pass
            last_resend = now
        time.sleep(0.1)  # 100ms 轮询

    try:
        client.loop_stop()
        client.disconnect()
    except Exception:
        pass

    return color_bgr, depth_2d


def capture_color_with_warmup():
    """连拍两次丢弃第一张 (刷新相机缓冲, 防止取到旧图)"""
    capture_color()  # 第一张丢弃, 刷新缓冲
    time.sleep(WARMUP_WAIT)
    return capture_color()


# ═══════════════════════════════════════════════════════════
#  YOLO 推理
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
#  远程 GPU 推理客户端 (MQTT)
# ═══════════════════════════════════════════════════════════

def _remote_infer(img_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF,
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


def detect_ab(model, img_bgr, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, strict_ab=True, min_boxes=1):
    """YOLO 检测 a/b, 返回 (best_dict, annotated) 或 None

    strict_ab=True: 严格要求 a 和 b 同时存在 (角度纠偏用)
    strict_ab=False: a/b 缺失时回退到 top2 (左右纠偏用) 或 single (前后纠偏用)
    min_boxes: strict_ab=False 回退时最少需要的检测框数量 (FB=1, LR=2)

    支持 REMOTE_INFER 模式: 通过 MQTT 请求远程 GPU 推理

    ★ 最低置信度过滤: conf < YOLO_MIN_CONF 的检测框直接丢弃, 防止围栏/背景误检

    ★ 辉羲 RPU 推理: 当 model 为 RhinoInfer 实例时, 直接调用芯片推理
    """
    # ── 辉羲 RPU 芯片推理路径 ──
    if USE_RHINO_INFER and isinstance(model, RhinoInfer):
        return model.detect_ab(img_bgr, conf_threshold=conf,
                               strict_ab=strict_ab, min_boxes=min_boxes)

    # 多阈值回退: 0.25 → 0.15 → 0.10 (不再降到 0.05/0.01, 避免低置信度误检)
    conf_thresholds = []
    for c in [conf, 0.15, YOLO_MIN_CONF]:
        if c >= YOLO_MIN_CONF and c not in [x for x in conf_thresholds]:
            conf_thresholds.append(c)
    if YOLO_MIN_CONF not in conf_thresholds:
        conf_thresholds.append(YOLO_MIN_CONF)

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
                # 调试: 打印所有检测到的框
                if all_boxes:
                    box_info = ", ".join([f"{b['cls']}={b['conf']:.2f}@({b['cx']:.0f},{b['cy']:.0f})" for b in all_boxes])
                    print(f"  [检测 conf={try_conf}] {len(all_boxes)}框: {box_info}")
                else:
                    print(f"  [检测 conf={try_conf}] 0框")
                if "a" in best and "b" in best:
                    return best, annotated
                continue
            else:
                # 远程失败
                err = resp.get("error", "未知")
                print(f"  [远程推理] 失败: {err}")
                if REMOTE_INFER_FALLBACK and model is not None:
                    print(f"  [远程推理] 回退到本地 CPU 推理")
                else:
                    return None, last_annotated

        # ===== 本地推理 (默认, 或远程失败回退) =====
        results = model(img_bgr, imgsz=imgsz, conf=try_conf, verbose=False)
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

        last_annotated = annotated
        last_all_boxes = all_boxes

        # 调试: 打印所有检测到的框
        if all_boxes:
            box_info = ", ".join([f"{b['cls']}={b['conf']:.2f}@({b['cx']:.0f},{b['cy']:.0f})" for b in all_boxes])
            print(f"  [检测 conf={try_conf}] {len(all_boxes)}框: {box_info}")
        else:
            print(f"  [检测 conf={try_conf}] 0框")

        if "a" in best and "b" in best:
            return best, annotated

    # strict_ab=True 位置回退: 2个高置信度框水平排列时, 按x坐标左=a右=b
    # 几何约束放宽: y_diff < max(avg_h * 2.0, 60) 防止小框被误拒
    if strict_ab and last_all_boxes and len(last_all_boxes) >= 2:
        sorted_by_conf = sorted(last_all_boxes, key=lambda b: b["conf"], reverse=True)
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
                print(f"  [位置回退] 2个高置信度框水平排列 (conf={top2[0]['conf']:.2f}/{top2[1]['conf']:.2f}, "
                      f"y_diff={y_diff:.0f}<{y_threshold:.0f}, x_dist={x_dist:.0f}>30), 左=a 右=b")
                return {"a": dict(left, cls="a"), "b": dict(right, cls="b")}, last_annotated

    # 回退: topN / single (仅 strict_ab=False 时)
    if not strict_ab and last_all_boxes and len(last_all_boxes) >= min_boxes:
        sorted_boxes = sorted(last_all_boxes, key=lambda b: b["conf"], reverse=True)
        if len(sorted_boxes) >= 2:
            top2 = sorted_boxes[:2]
            return {"a": top2[0], "b": top2[1]}, last_annotated
        else:
            only = sorted_boxes[0]
            print(f"  [检测回退] 仅检测到1个目标 ({only['cls']} conf={only['conf']:.2f}), a/b 复用同一框")
            return {"a": only, "b": only}, last_annotated

    return None, last_annotated


def get_depth_at_point(depth_2d, cx, cy, r=5):
    """取深度图中指定坐标附近的深度值 (逐步扩大搜索半径, 中位数, mm)"""
    h, w = depth_2d.shape
    for radius in (r, r*2, r*4, r*8, r*16):
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        region = depth_2d[y0:y1, x0:x1]
        valid = region[(region > 0) & (region < 10000)]
        if len(valid) >= 5:
            return float(np.median(valid))
    return 0.0


# ═══════════════════════════════════════════════════════════
#  机器人控制
# ═══════════════════════════════════════════════════════════

def setup_minth():
    if MINTH_DIR not in sys.path:
        sys.path.insert(0, MINTH_DIR)
    import minth
    return minth.G2(broker=MQTT_BROKER, port=MQTT_PORT, timeout=60)


def move_chassis(g2, dx_m=0.0, dy_m=0.0, yaw_rad=0.0):
    """底盘相对运动"""
    return g2._send_and_wait("go_rel", {"x": dx_m, "y": dy_m, "yaw_rad": yaw_rad})


# ═══════════════════════════════════════════════════════════
#  绘制
# ═══════════════════════════════════════════════════════════

def draw_top_bar(img, lines):
    """在图像顶部添加数据栏 (支持 1~N 行)"""
    h, w = img.shape[:2]
    n = max(len(lines), 1)
    bar_h = 24 * n + 12
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(bar, line, (12, 24 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2)
    return np.vstack([bar, img])


def draw_geometry(img, best, mid_x, mid_y, iteration, info_lines):
    """绘制 a/b 连线、中点、中垂线、图像中心竖直线"""
    h, w = img.shape[:2]
    a_cx, a_cy = best["a"]["cx"], best["a"]["cy"]
    b_cx, b_cy = best["b"]["cx"], best["b"]["cy"]

    cv2.line(img, (int(a_cx), int(a_cy)), (int(b_cx), int(b_cy)), COLOR_LINE_AB, 2)
    cv2.circle(img, (int(mid_x), int(mid_y)), 6, COLOR_MID, -1)

    # 中垂线
    dx, dy = b_cx - a_cx, b_cy - a_cy
    length = max(np.sqrt(dx*dx + dy*dy), 1e-6)
    perp_x, perp_y = -dy / length, dx / length
    L = max(h, w)
    cv2.line(img, (int(mid_x - perp_x * L), int(mid_y - perp_y * L)),
             (int(mid_x + perp_x * L), int(mid_y + perp_y * L)), COLOR_PERP, 1)

    # 图像中心竖直线
    cv2.line(img, (LR_TARGET_X, 0), (LR_TARGET_X, h), COLOR_CENTER, 1)

    return draw_top_bar(img, info_lines)


def _save_step_final(output_dir, prefix, iteration, img, annotated, info_lines, status=""):
    """保存纠偏阶段最终结果图 (不论成功失败都输出)

    Args:
        output_dir: 输出目录
        prefix: 文件名前缀, 如 "01_fb"
        iteration: 当前迭代次数
        img: 原始彩色图 (annotated 为 None 时使用)
        annotated: YOLO 标注图 (优先使用)
        info_lines: 顶部信息栏文本列表 (最多2行)
        status: 状态后缀, 如 "no_ab", "depth_invalid", "max_iter" (空字符串表示成功收敛)

    Returns:
        保存的文件路径, 或 None (无图像可保存)
    """
    ts = time.strftime("%H%M%S")
    base = annotated if annotated is not None else img
    if base is None:
        print(f"[保存] ⚠ 无图像可保存 ({prefix} iter{iteration})")
        return None
    final_img = base.copy()
    if info_lines:
        final_img = draw_top_bar(final_img, info_lines)
    status_suffix = f"_{status}" if status else ""
    final_path = os.path.join(output_dir, f"{prefix}_iter{iteration}_{ts}{status_suffix}.jpg")
    cv2.imwrite(final_path, final_img)
    print(f"[保存] 结果图已保存: {final_path}")
    return final_path


# ═══════════════════════════════════════════════════════════
#  纠偏步骤 1: 前后
# ═══════════════════════════════════════════════════════════

def step_fb_correct(model, g2, target_depth, output_dir, dry_run):
    """前后纠偏: 通过深度图对齐目标距离"""
    print(f"\n{'='*60}")
    print(f"[步骤 1/3] 前后纠偏 (目标深度: {target_depth}mm)")
    print(f"{'='*60}")

    curr_gain = FB_GAIN
    prev_delta = None
    prev_move_m = None
    total_move = 0.0
    last_move_dir = 0
    last_delta = None
    last_avg_d = None           # 上一次平均深度 (用于检测旧图缓冲)
    last_move_m_abs = 0.0       # 上一次实际移动距离绝对值 (m)

    for iteration in range(1, FB_MAX_ITER + 1):
        print(f"\n--- 前后纠偏 迭代 {iteration}/{FB_MAX_ITER} ---")

        # 移动后预拍照刷新相机缓冲 (第2次迭代起, 防止取到移动前的旧图)
        if iteration > 1:
            print(f"[FB-{iteration}] 预拍照刷新缓冲...")
            capture_color_and_depth()
            time.sleep(WARMUP_WAIT)

        # 拍照 (彩色 + 深度, 连拍两次)
        color_img, depth_2d = None, None
        for attempt in range(3):
            print(f"[FB-{iteration}] 拍照中... (第{attempt+1}次)")
            color_img, depth_2d = capture_color_and_depth()
            if color_img is not None and depth_2d is not None:
                break
            time.sleep(1.0)

        if color_img is None or depth_2d is None:
            print(f"[FB-{iteration}] ✗ 拍照失败")
            return {"success": False, "step": "fb", "reason": "拍照失败", "color_img": None}

        # YOLO 检测
        best, annotated = detect_ab(model, color_img, strict_ab=False)
        if best is None:
            print(f"[FB-{iteration}] ✗ 未检测到 a/b")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  STATUS: NO_AB_DETECTED",
                              f"reason: a/b not found in image"], status="no_ab")
            return {"success": False, "step": "fb", "reason": "未检测到 a/b", "color_img": color_img}

        # 深度值
        a_d = get_depth_at_point(depth_2d, int(best["a"]["cx"]), int(best["a"]["cy"]))
        b_d = get_depth_at_point(depth_2d, int(best["b"]["cx"]), int(best["b"]["cy"]))
        avg_d = (a_d + b_d) / 2
        delta = avg_d - target_depth

        print(f"[FB-{iteration}] a={a_d:.0f}mm b={b_d:.0f}mm avg={avg_d:.0f}mm delta={delta:+.0f}mm")

        # 每次拍照都保存标注图
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

        if a_d == 0 or b_d == 0:
            print(f"[FB-{iteration}] ✗ 深度值无效")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  STATUS: DEPTH_INVALID",
                              f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm (invalid)"],
                             status="depth_invalid")
            return {"success": False, "step": "fb", "reason": "深度无效", "color_img": color_img}

        # 深度一致性检查: a/b 深度差超过 FB_MAX_DEPTH_DIFF 视为误检 (防止低置信度误检导致深度异常)
        FB_MAX_DEPTH_DIFF = 300  # mm, a/b 最大允许深度差
        depth_diff = abs(a_d - b_d)
        if depth_diff > FB_MAX_DEPTH_DIFF:
            print(f"[FB-{iteration}] ⚠ a/b 深度差={depth_diff:.0f}mm > {FB_MAX_DEPTH_DIFF}mm, 疑似误检 (a={a_d:.0f} b={b_d:.0f})")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  STATUS: DEPTH_INCONSISTENT",
                              f"a:{a_d:.0f}mm b:{b_d:.0f}mm diff:{depth_diff:.0f}mm (false detect?)"],
                             status="depth_inconsistent")
            return {"success": False, "step": "fb", "reason": f"a/b深度差过大({depth_diff:.0f}mm),疑似误检", "color_img": color_img}

        # 最小深度安全检查 (防止撞架)
        if avg_d < FB_MIN_DEPTH:
            print(f"[FB-{iteration}] ⚠⚠ 安全终止: 深度过近 avg={avg_d:.0f}mm < {FB_MIN_DEPTH}mm, 立即停止!")
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  STATUS: TOO_CLOSE_ABORT",
                              f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm < {FB_MIN_DEPTH}mm"],
                             status="too_close_abort")
            return {"success": False, "step": "fb", "reason": f"深度过近({avg_d:.0f}mm),防止撞架", "color_img": color_img}

        # 安全检查
        curr_dir = 1 if delta > 0 else -1
        if last_move_dir != 0 and last_delta is not None:
            # 1) 偏差异常增大 → 终止 (方向错误或外部干扰)
            if abs(delta) - abs(last_delta) > 30:
                print(f"[FB-{iteration}] ⚠⚠ 安全终止: 偏差异常增大")
                _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                                 [f"FB iter:{iteration}  STATUS: SAFETY_ABORT",
                                  f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm delta:{delta:+.0f}mm"],
                                 status="safety_abort")
                return {"success": False, "step": "fb", "reason": "安全终止", "color_img": color_img}
            # 2) 移动后深度几乎不变 → 疑似取到旧图缓冲 → 终止 (防止基于旧图重复移动撞架)
            if abs(avg_d - last_avg_d) < 10 and last_move_m_abs > 0.05:
                print(f"[FB-{iteration}] ⚠⚠ 安全终止: 移动{last_move_m_abs*1000:.0f}mm后深度几乎不变 "
                      f"(avg {last_avg_d:.0f}→{avg_d:.0f}mm), 疑似取到旧图")
                _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                                 [f"FB iter:{iteration}  STATUS: STALE_BUFFER_ABORT",
                                  f"moved {last_move_m_abs*1000:.0f}mm but depth {last_avg_d:.0f}→{avg_d:.0f}mm (stale buffer?)"],
                                 status="stale_buffer_abort")
                return {"success": False, "step": "fb", "reason": "疑似旧图缓冲,安全终止", "color_img": color_img}

        # 收敛判断
        if abs(delta) < FB_THRESHOLD:
            print(f"[FB-{iteration}] ✓ 收敛 (|{delta:.0f}| < {FB_THRESHOLD})")
            # 保存结果图
            ts = time.strftime("%H%M%S")
            final_img = annotated.copy()
            cv2.circle(final_img, (int(best["a"]["cx"]), int(best["a"]["cy"])), 4, (0,0,255), -1)
            cv2.circle(final_img, (int(best["b"]["cx"]), int(best["b"]["cy"])), 4, (0,0,255), -1)
            cv2.putText(final_img, f"{a_d:.0f}mm", (int(best["a"]["cx"])+8, int(best["a"]["cy"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            cv2.putText(final_img, f"{b_d:.0f}mm", (int(best["b"]["cx"])+8, int(best["b"]["cy"])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
            info1 = f"FB iter:{iteration}  a:{a_d:.0f}mm  b:{b_d:.0f}mm"
            info2 = f"avg:{avg_d:.0f}mm  target:{target_depth}mm  delta:{delta:+.0f}mm"
            final_img = draw_top_bar(final_img, [info1, info2])
            final_path = os.path.join(output_dir, f"01_fb_final_iter{iteration}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            print(f"[FB-{iteration}] 结果图已保存: {final_path}")
            return {"success": True, "step": "fb", "converged": True,
                    "a_depth": a_d, "b_depth": b_d, "avg_depth": avg_d,
                    "color_img": color_img}

        # 自适应增益
        if prev_delta is not None and prev_move_m is not None:
            actual_change = prev_delta - delta
            expected_change = prev_move_m * 1000.0  # mm
            if abs(expected_change) > 1.0:
                ratio = actual_change / expected_change
                if abs(ratio) > 0.1:
                    ideal_gain = 1.0 / ratio
                    new_gain = 0.5 * curr_gain + 0.5 * ideal_gain
                    new_gain = max(0.5, min(3.0, new_gain))
                    print(f"[FB-{iteration}] 自适应: ratio={ratio:.2f} gain {curr_gain:.2f}→{new_gain:.2f}")
                    curr_gain = new_gain

        # 计算移动 (带增益)
        move_m = delta / 1000.0 * curr_gain
        if abs(move_m) > FB_MAX_SINGLE_MOVE:
            move_m = FB_MAX_SINGLE_MOVE if move_m > 0 else -FB_MAX_SINGLE_MOVE

        print(f"[FB-{iteration}] 移动: {move_m*1000:+.0f}mm ({'前进' if move_m > 0 else '后退'}) [gain={curr_gain:.2f}]")

        if dry_run:
            print(f"[FB-{iteration}] (DRY RUN)")
            prev_delta = delta
            prev_move_m = move_m
            continue

        # 小距离移动时预移动 (克服静摩擦: 先反向再正向, 净位移不变)
        if FB_PRE_MOVE and abs(move_m) < FB_PRE_MOVE_THRESHOLD:
            pre_m = FB_PRE_MOVE_M * (-1 if move_m > 0 else 1)
            print(f"[FB-{iteration}] 小距离预热: 先{'后退' if pre_m < 0 else '前进'}{abs(pre_m)*1000:.0f}mm")
            move_chassis(g2, dx_m=pre_m)
            time.sleep(FB_SETTLE_TIME)
            ok = move_chassis(g2, dx_m=move_m - pre_m)
        else:
            ok = move_chassis(g2, dx_m=move_m)

        if not ok:
            _save_step_final(output_dir, "01_fb", iteration, color_img, annotated,
                             [f"FB iter:{iteration}  STATUS: MOVE_FAILED",
                              f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm delta:{delta:+.0f}mm"],
                             status="move_failed")
            return {"success": False, "step": "fb", "reason": "移动失败", "color_img": color_img}

        total_move += move_m
        last_move_dir = curr_dir
        last_delta = delta
        last_avg_d = avg_d               # 记录本次深度, 用于下次检测旧图
        last_move_m_abs = abs(move_m)    # 记录本次移动距离, 用于下次检测旧图
        prev_delta = delta
        prev_move_m = move_m
        time.sleep(FB_SETTLE_TIME)

    _save_step_final(output_dir, "01_fb", FB_MAX_ITER, color_img, annotated,
                     [f"FB iter:{FB_MAX_ITER}  STATUS: MAX_ITER_NOT_CONVERGED",
                      f"a:{a_d:.0f}mm b:{b_d:.0f}mm avg:{avg_d:.0f}mm delta:{delta:+.0f}mm"],
                     status="max_iter")
    return {"success": True, "step": "fb", "converged": False, "color_img": color_img}


# ═══════════════════════════════════════════════════════════
#  纠偏步骤 2: 角度
# ═══════════════════════════════════════════════════════════

def step_yaw_correct(model, g2, output_dir, dry_run, reuse_img=None):
    """角度纠偏: 通过 a/b 连线斜率对齐底盘 Yaw (自适应增益)"""
    print(f"\n{'='*60}")
    print(f"[步骤 2/3] 角度纠偏 (目标: angle → 0°)")
    print(f"{'='*60}")

    curr_gain = YAW_GAIN
    prev_angle = None
    prev_yaw_deg = None
    total_yaw = 0.0

    for iteration in range(1, YAW_MAX_ITER + 1):
        print(f"\n--- 角度纠偏 迭代 {iteration}/{YAW_MAX_ITER} ---")

        # 拍照: 第1次迭代可复用前一步的彩色图, 失败后重拍
        best, annotated, img = None, None, None
        new_photo_count = 0
        max_new_photos = 3  # 复用图失败后最多重拍3次
        for attempt in range(max_new_photos + 1):
            if attempt == 0 and iteration == 1 and reuse_img is not None:
                print(f"[YAW-{iteration}] 复用上一步彩色图")
                img = reuse_img
            else:
                new_photo_count += 1
                print(f"[YAW-{iteration}] 拍照中... (第{new_photo_count}次)")
                img = capture_color()

            # YOLO 检测 (严格 a+b)
            best, annotated = detect_ab(model, img, strict_ab=True)
            if best is not None:
                break
            print(f"[YAW-{iteration}] ✗ 未同时检测到 a/b, 重试...")
            time.sleep(0.3)

        if best is None:
            print(f"[YAW-{iteration}] ✗ 多次拍照仍未检测到 a/b")
            _save_step_final(output_dir, "02_yaw", iteration, img, annotated,
                             [f"YAW iter:{iteration}  STATUS: NO_AB_DETECTED",
                              f"reason: a and b not detected simultaneously"], status="no_ab")
            return {"success": False, "step": "yaw", "reason": "未检测到 a/b", "color_img": img}

        a_cx, a_cy = best["a"]["cx"], best["a"]["cy"]
        b_cx, b_cy = best["b"]["cx"], best["b"]["cy"]
        dx = b_cx - a_cx
        slope = (b_cy - a_cy) / dx if abs(dx) > 1e-6 else float("inf")
        angle = np.degrees(np.arctan(slope)) if abs(slope) < 1e6 else 90.0

        mid_x = (a_cx + b_cx) / 2
        mid_y = (a_cy + b_cy) / 2

        print(f"[YAW-{iteration}] slope={slope:+.4f} angle={angle:+.2f}° gain={curr_gain:.2f}")

        # 每次拍照都保存标注图
        ts = time.strftime("%H%M%S")
        iter_img = draw_geometry(annotated.copy(), best, mid_x, mid_y, iteration,
                                 [f"YAW iter:{iteration} slope:{slope:+.4f} angle:{angle:+.2f}deg gain:{curr_gain:.2f}"])
        cv2.imwrite(os.path.join(output_dir, f"02_yaw_iter{iteration}_{ts}.jpg"), iter_img)

        # 收敛判断
        if abs(angle) < YAW_THRESHOLD_DEG:
            print(f"[YAW-{iteration}] ✓ 收敛 (|{angle:.2f}| < {YAW_THRESHOLD_DEG})")
            # 保存结果图
            ts = time.strftime("%H%M%S")
            final_img = draw_geometry(annotated.copy(), best, mid_x, mid_y, iteration,
                                      [f"YAW iter:{iteration}  slope:{slope:+.4f}  angle:{angle:+.2f}deg",
                                       f"gain:{curr_gain:.2f}  target:0deg"])
            final_path = os.path.join(output_dir, f"02_yaw_final_iter{iteration}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            print(f"[YAW-{iteration}] 结果图已保存: {final_path}")
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

        # 预旋转: 小角度旋转时先反向大角度旋转, 克服底盘静摩擦
        if YAW_PRE_ROTATE and abs(delta_yaw_deg) < YAW_PRE_ROTATE_THRESHOLD:
            pre_deg = YAW_PRE_ROTATE_DEG * (-1 if delta_yaw_deg > 0 else 1)
            print(f"[YAW-{iteration}] [预旋转] 先转 {pre_deg:+.1f}° 再补偿 (克服静摩擦)")
            move_chassis(g2, yaw_rad=np.radians(pre_deg))
            time.sleep(YAW_SETTLE_TIME)
            # 补偿: 正向旋转 = 预旋转角度 + 实际旋转量
            delta_yaw_rad = delta_yaw_rad - np.radians(pre_deg)

        ok = move_chassis(g2, yaw_rad=delta_yaw_rad)
        if not ok:
            _save_step_final(output_dir, "02_yaw", iteration, img, annotated,
                             [f"YAW iter:{iteration}  STATUS: ROTATE_FAILED",
                              f"slope:{slope:+.4f} angle:{angle:+.2f}deg delta_yaw:{delta_yaw_deg:+.2f}deg"],
                             status="rotate_failed")
            return {"success": False, "step": "yaw", "reason": "旋转失败", "color_img": img}

        total_yaw += delta_yaw_rad
        prev_angle = angle
        prev_yaw_deg = delta_yaw_deg
        time.sleep(YAW_SETTLE_TIME)

    _save_step_final(output_dir, "02_yaw", YAW_MAX_ITER, img, annotated,
                     [f"YAW iter:{YAW_MAX_ITER}  STATUS: MAX_ITER_NOT_CONVERGED",
                      f"slope:{slope:+.4f} angle:{angle:+.2f}deg gain:{curr_gain:.2f}"],
                     status="max_iter")
    return {"success": True, "step": "yaw", "converged": False, "color_img": img}


# ═══════════════════════════════════════════════════════════
#  纠偏步骤 3: 左右
# ═══════════════════════════════════════════════════════════

def step_lr_correct(model, g2, output_dir, dry_run, reuse_img=None):
    """左右纠偏: 通过 a/b 中点 x 对齐图像中心"""
    print(f"\n{'='*60}")
    print(f"[步骤 3/3] 左右纠偏 (目标: mid_x → {LR_TARGET_X}px)")
    print(f"{'='*60}")

    curr_gain = LR_GAIN
    prev_delta_px = None
    prev_move_m = None
    total_move = 0.0

    for iteration in range(1, LR_MAX_ITER + 1):
        print(f"\n--- 左右纠偏 迭代 {iteration}/{LR_MAX_ITER} ---")

        # 拍照: 第1次迭代可复用前一步的彩色图
        if iteration == 1 and reuse_img is not None:
            print(f"[LR-{iteration}] 复用上一步彩色图")
            img = reuse_img
        else:
            img = capture_color()

        # YOLO 检测 (允许 top2 回退, 但至少需要2个框才能算中点)
        best, annotated = detect_ab(model, img, strict_ab=False, min_boxes=2)
        if best is None:
            print(f"[LR-{iteration}] ✗ 未检测到 a/b")
            _save_step_final(output_dir, "03_lr", iteration, img, annotated,
                             [f"LR iter:{iteration}  STATUS: NO_AB_DETECTED",
                              f"reason: a/b not found in image"], status="no_ab")
            return {"success": False, "step": "lr", "reason": "未检测到 a/b", "color_img": img}

        mid_x = (best["a"]["cx"] + best["b"]["cx"]) / 2
        mid_y = (best["a"]["cy"] + best["b"]["cy"]) / 2
        delta_px = mid_x - LR_TARGET_X

        print(f"[LR-{iteration}] mid_x={mid_x:.1f} delta={delta_px:+.1f}px")

        # 每次拍照都保存标注图
        ts = time.strftime("%H%M%S")
        iter_img = draw_geometry(annotated.copy(), best, mid_x, mid_y, iteration,
                                 [f"LR iter:{iteration} mid_x:{mid_x:.1f} delta:{delta_px:+.1f}px gain:{curr_gain:.2f}"])
        cv2.imwrite(os.path.join(output_dir, f"03_lr_iter{iteration}_{ts}.jpg"), iter_img)

        # 收敛判断
        if abs(delta_px) < LR_THRESHOLD:
            print(f"[LR-{iteration}] ✓ 收敛 (|{delta_px:.1f}| < {LR_THRESHOLD})")
            # 保存结果图
            ts = time.strftime("%H%M%S")
            final_img = draw_geometry(annotated.copy(), best, mid_x, mid_y, iteration,
                                      [f"LR iter:{iteration}  mid_x:{mid_x:.1f}  delta:{delta_px:+.1f}px",
                                       f"move:{(-delta_px*PX_TO_METER)*1000:+.1f}mm  target:{LR_TARGET_X}px"])
            final_path = os.path.join(output_dir, f"03_lr_final_iter{iteration}_{ts}.jpg")
            cv2.imwrite(final_path, final_img)
            print(f"[LR-{iteration}] 结果图已保存: {final_path}")
            return {"success": True, "step": "lr", "converged": True,
                    "final_mid_x": mid_x, "final_delta_px": delta_px, "color_img": img}

        # 自适应增益
        if prev_delta_px is not None and prev_move_m is not None:
            actual_change = prev_delta_px - delta_px  # px变化
            expected_change = -prev_move_m / PX_TO_METER  # 预期px变化
            if abs(expected_change) > 0.5:
                ratio = actual_change / expected_change
                if abs(ratio) > 0.01:
                    ideal_gain = 1.0 / ratio
                    new_gain = 0.5 * curr_gain + 0.5 * ideal_gain
                    new_gain = max(0.5, min(10.0, new_gain))
                    print(f"[LR-{iteration}] 自适应: ratio={ratio:.2f} gain {curr_gain:.2f}→{new_gain:.2f}")
                    curr_gain = new_gain

        # 计算移动量 (带增益补偿)
        move_m = -delta_px * PX_TO_METER * curr_gain
        print(f"[LR-{iteration}] 移动: {move_m*1000:+.1f}mm ({'左' if move_m > 0 else '右'}) [gain={curr_gain:.2f}]")

        if dry_run:
            print(f"[LR-{iteration}] (DRY RUN)")
            prev_delta_px = delta_px
            prev_move_m = move_m
            continue

        # 小距离移动时预横移 (克服静摩擦: 先反向再正向, 净位移不变)
        if LR_PRE_MOVE and abs(move_m) < LR_PRE_MOVE_THRESHOLD:
            pre_m = LR_PRE_MOVE_M * (-1 if move_m > 0 else 1)
            print(f"[LR-{iteration}] 小距离预热: 先{'右移' if pre_m < 0 else '左移'}{abs(pre_m)*1000:.0f}mm")
            move_chassis(g2, dy_m=pre_m)
            time.sleep(LR_SETTLE_TIME)
            ok = move_chassis(g2, dy_m=move_m - pre_m)
        else:
            ok = move_chassis(g2, dy_m=move_m)

        if not ok:
            _save_step_final(output_dir, "03_lr", iteration, img, annotated,
                             [f"LR iter:{iteration}  STATUS: MOVE_FAILED",
                              f"mid_x:{mid_x:.1f} delta:{delta_px:+.1f}px move:{move_m*1000:+.1f}mm"],
                             status="move_failed")
            return {"success": False, "step": "lr", "reason": "移动失败", "color_img": img}

        total_move += move_m
        prev_delta_px = delta_px
        prev_move_m = move_m
        time.sleep(LR_SETTLE_TIME)

    _save_step_final(output_dir, "03_lr", LR_MAX_ITER, img, annotated,
                     [f"LR iter:{LR_MAX_ITER}  STATUS: MAX_ITER_NOT_CONVERGED",
                      f"mid_x:{mid_x:.1f} delta:{delta_px:+.1f}px target:{LR_TARGET_X}px"],
                     status="max_iter")
    return {"success": True, "step": "lr", "converged": False, "color_img": img}


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="底盘综合纠偏: 前后 → 角度 → 左右")
    parser.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH,
                        help=f"目标深度 mm (默认: {DEFAULT_TARGET_DEPTH})")
    parser.add_argument("--dry-run", action="store_true", help="只检测不移动")
    parser.add_argument("--skip-fb", action="store_true", help="跳过前后纠偏")
    parser.add_argument("--skip-yaw", action="store_true", help="跳过角度纠偏")
    parser.add_argument("--skip-lr", action="store_true", help="跳过左右纠偏")
    parser.add_argument("--remote-infer", action="store_true",
                        help="启用远程 GPU 推理 (MQTT, 需先启动 yolo_infer_server.py)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="远程推理失败时不回退到本地 CPU 推理")
    args = parser.parse_args()

    # 应用远程推理配置
    global REMOTE_INFER, REMOTE_INFER_FALLBACK
    if args.remote_infer:
        REMOTE_INFER = True
        print(f"[配置] 远程 GPU 推理: 启用")
    if args.no_fallback:
        REMOTE_INFER_FALLBACK = False

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total_start = time.time()

    print(f"{'='*60}")
    print(f"底盘综合纠偏总控程序")
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
    elif REMOTE_INFER:
        if REMOTE_INFER_FALLBACK:
            print(f"\n[初始化] 远程推理模式 + 本地回退, 加载本地模型: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
        else:
            print(f"\n[初始化] 远程推理模式 (无回退), 跳过本地模型加载")
            model = None
    else:
        print(f"\n[初始化] 加载 YOLO 模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)

    # 2. 连接机器人 (只连接一次)
    if args.dry_run:
        print(f"[初始化] DRY RUN 模式, 跳过机器人连接")
        g2 = None
    else:
        print(f"[初始化] 连接机器人...")
        g2 = setup_minth()
        print(f"[初始化] ✓ Minth 已就绪")

    results = {}
    abort = False  # 任一纠偏异常则终止后续步骤

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

    # 步骤 2: 角度纠偏 (前后纠偏异常则跳过)
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

    # 步骤 3: 左右纠偏 (前序纠偏异常则跳过)
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

    # 异常终止时立即释放机器人连接
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
    print(f"综合纠偏总结 (耗时 {elapsed:.1f}s)")
    print(f"{'='*60}")

    if "fb" in results:
        r = results["fb"]
        if r.get("converged"):
            print(f"  前后: ✓ 收敛  a={r.get('a_depth',0):.0f}mm b={r.get('b_depth',0):.0f}mm avg={r.get('avg_depth',0):.0f}mm")
        elif r.get("success"):
            print(f"  前后: ⚠ 未完全收敛")
        else:
            print(f"  前后: ✗ 失败 ({r.get('reason')})")

    if "yaw" in results:
        r = results["yaw"]
        if r.get("converged"):
            print(f"  角度: ✓ 收敛  angle={r.get('final_angle',0):+.2f}°")
        elif r.get("success"):
            print(f"  角度: ⚠ 未完全收敛")
        else:
            print(f"  角度: ✗ 失败 ({r.get('reason')})")

    if "lr" in results:
        r = results["lr"]
        if r.get("converged"):
            print(f"  左右: ✓ 收敛  mid_x={r.get('final_mid_x',0):.1f} delta={r.get('final_delta_px',0):+.1f}px")
        elif r.get("success"):
            print(f"  左右: ⚠ 未完全收敛")
        else:
            print(f"  左右: ✗ 失败 ({r.get('reason')})")

    print(f"{'='*60}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[中断] 用户中止")
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"✗ 程序异常崩溃: {type(e).__name__}: {e}")
        print(f"{'='*60}")
        traceback.print_exc()
        sys.exit(2)  # 退出码 2 = 崩溃 (区别于 0 = 正常完成)
    sys.exit(0)
