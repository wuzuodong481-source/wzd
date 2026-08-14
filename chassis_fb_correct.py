#!/usr/bin/env python3
"""
底盘前后纠偏程序

目标: 让 a/b 目标的平均深度对齐目标距离 (默认 700mm), 通过底盘 X 方向前后移动实现闭环纠偏。

原理:
  1. 拍 RGB 图 → YOLO 检测 a/b 坐标
  2. 同时拍深度图 → 读取 a/b 位置的 uint16 深度值 (单位 mm)
  3. 计算 a/b 平均深度与目标距离的偏差
  4. 若 |偏差| > 阈值, 底盘前后移动: dx = -(当前深度 - 目标深度) / 1000
     (当前深度 > 目标 → 需前进 → dx 为正)
  5. 等待稳定后重新拍照验证, 直到收敛或达到最大迭代次数

深度图获取方式:
  - 通过 MQTT 发送 save_photo 命令, 同时保存 kHeadColor 和 kHeadDepth
  - 读取 /data/wxf/wxf0721/images/ 目录下的 .jpg (彩色) 和 .raw (深度 uint16)

用法:
  python chassis_fb_correct.py
  python chassis_fb_correct.py --target-depth 700
  python chassis_fb_correct.py --max-iter 5
  python chassis_fb_correct.py --dry-run    # 只检测不移动
"""
import argparse
import os
import sys
import time
import json
import glob
import base64
import threading
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== 路径配置 (可后续修改) =====================
MODEL_PATH = "/data/wzd/best_new.pt"
OUTPUT_DIR = "/data/wzd/correction_fb"
GDK_SERVICES_DIR = "/data/wxf/wxf0721/services"
MINTH_DIR = "/data/wxf/wxf0721/runtime"
# 相机服务保存图片的目录
IMAGE_SAVE_DIR = "/data/wxf/wxf0721/images"

# ===================== 纠偏参数 =====================
# 目标深度 (mm)
DEFAULT_TARGET_DEPTH = 700
# 收敛阈值 (mm), |delta_mm| < 阈值 视为收敛
DEFAULT_THRESHOLD = 20
# 最大迭代次数
DEFAULT_MAX_ITER = 5
# 每次移动后等待稳定时间 (秒)
SETTLE_TIME = 3.0
# 单次最大移动距离 (米), 防止过冲
MAX_SINGLE_MOVE = 0.3  # 300mm
# MQTT 配置
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# ===================== 远程 GPU 推理配置 =====================
REMOTE_INFER = False                      # 是否启用远程推理 (默认本地)
REMOTE_INFER_TOPIC_REQ = "/yolo/infer/request"
REMOTE_INFER_TOPIC_RSP = "/yolo/infer/response"
REMOTE_INFER_TIMEOUT = 5.0                # 单次推理超时 s
REMOTE_INFER_FALLBACK = True              # 远程失败时回退到本地 CPU 推理

# ===================== 绘制参数 =====================
COLOR_LINE_AB = (0, 255, 0)
COLOR_MID = (0, 0, 255)
COLOR_BAR_BG = (0, 0, 0)
COLOR_BAR_TEXT = (255, 255, 255)


# ═══════════════════════════════════════════════════════════
#  通过 MQTT 拍照并保存 (彩色 + 深度)
# ═══════════════════════════════════════════════════════════

def _send_save_photo(broker, port, timeout=15.0):
    """发送一次 save_photo 命令并等待完成

    返回: True/False
    """
    import paho.mqtt.client as mqtt

    done = {"flag": False}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            client.subscribe("/humanoid/commands/done", qos=2)

    def on_message(client, userdata, msg):
        if msg.topic == "/humanoid/commands/done":
            try:
                payload = json.loads(msg.payload.decode("utf-8"))
                if payload.get("cmd") == "save_photo":
                    done["flag"] = True
            except Exception:
                pass

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)
    client.loop_start()

    cmd = json.dumps({"command": "save_photo", "cameras": ["kHeadColor", "kHeadDepth"]})
    client.publish("/humanoid/camera/control", cmd, qos=2)

    t_start = time.time()
    while not done["flag"] and time.time() - t_start < timeout:
        time.sleep(0.3)

    client.loop_stop()
    client.disconnect()
    return done["flag"]


def _read_latest_color_and_depth(existing_color, existing_depth, tried_depth=None):
    """从 IMAGE_SAVE_DIR 中查找新增的彩色图和深度图并读取

    Args:
        existing_color: 之前的 color 文件集合
        existing_depth: 之前的 depth 文件集合
        tried_depth: 已尝试过但异常的 depth 文件集合 (跳过这些)

    返回: (color_bgr, depth_2d, color_path, depth_path) 或 (None, None, None, None)
    """
    if tried_depth is None:
        tried_depth = set()
    new_color = sorted(set(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadColor_*.jpg"))) - existing_color)
    new_depth = sorted(set(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadDepth_raw_*.raw"))) - existing_depth - tried_depth)

    if not new_color or not new_depth:
        return None, None, None, None

    color_path = new_color[-1]
    depth_path = new_depth[-1]

    # 验证时间戳一致
    color_ts = os.path.basename(color_path).replace("kHeadColor_", "").replace(".jpg", "")
    depth_ts = os.path.basename(depth_path).replace("kHeadDepth_raw_", "").replace(".raw", "")
    if color_ts != depth_ts:
        print(f"[相机] ⚠ 彩色图({color_ts})与深度图({depth_ts})时间戳不一致")

    color_bgr = cv2.imread(color_path)
    if color_bgr is None:
        return None, None, None, None

    with open(depth_path, "rb") as f:
        raw_data = f.read()
    depth_uint16 = np.frombuffer(raw_data, dtype=np.uint16)

    h, w = color_bgr.shape[:2]
    expected = w * h
    if depth_uint16.size == expected:
        depth_2d = depth_uint16.reshape((h, w))
    else:
        for try_w, try_h in [(640, 400), (640, 480), (320, 240)]:
            if depth_uint16.size == try_w * try_h:
                depth_2d = depth_uint16.reshape((try_h, try_w))
                depth_2d = cv2.resize(depth_2d, (w, h), interpolation=cv2.INTER_NEAREST)
                break
        else:
            print(f"[相机] 深度图分辨率不匹配: pixels={depth_uint16.size}, expected={expected}")
            return None, None, None, None

    return color_bgr, depth_2d, color_path, depth_path


def capture_color_and_depth(broker=MQTT_BROKER, port=MQTT_PORT, timeout=15.0):
    """单次拍照获取彩色+深度图 (优化: 轮询文件出现, 省去 done 等待)

    返回: (color_bgr, depth_uint16) 或 (None, None)
    """
    existing_c = set(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadColor_*.jpg")))
    existing_d = set(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadDepth_raw_*.raw")))

    # 发送 save_photo 命令 (不等待 done, 直接轮询文件)
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

    # 轮询等待新文件出现 (一旦两个文件都到位立即返回)
    t_start = time.time()
    color_bgr, depth_2d = None, None
    tried_depth = set()  # 已尝试过的异常深度文件
    while time.time() - t_start < timeout:
        color_bgr, depth_2d, _, _ = _read_latest_color_and_depth(existing_c, existing_d, tried_depth)
        if color_bgr is not None and depth_2d is not None:
            break
        # depth 异常: 标记最新文件以跳过, 等待新文件
        if depth_2d is None:
            depth_files = sorted(glob.glob(os.path.join(IMAGE_SAVE_DIR, "kHeadDepth_raw_*.raw")))
            new_d_now = [f for f in depth_files if f not in existing_d and f not in tried_depth]
            if new_d_now:
                tried_depth.add(new_d_now[-1])
        time.sleep(0.1)  # 100ms 轮询

    try:
        client.loop_stop()
        client.disconnect()
    except Exception:
        pass

    if color_bgr is not None and depth_2d is not None:
        print(f"[相机] 彩色图: {color_bgr.shape}, 深度图: {depth_2d.shape}")
    return color_bgr, depth_2d


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
      2. 若始终只有 a 或只有 b, 回退到取置信度最高的 2 个框作为中点

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
#  深度值提取
# ═══════════════════════════════════════════════════════════

def get_depth_at_point(depth_2d, cx, cy, r=5):
    """在深度图中取指定坐标附近的深度值 (取 5x5 邻域中位数, 单位 mm)

    过滤掉 0 值 (无效深度) 和异常大值 (>10000mm)
    """
    h, w = depth_2d.shape
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    region = depth_2d[y0:y1, x0:x1]
    # 过滤无效值
    valid = region[(region > 0) & (region < 10000)]
    if len(valid) > 0:
        return float(np.median(valid))
    return 0.0


# ═══════════════════════════════════════════════════════════
#  机器人控制
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
#  图像绘制
# ═══════════════════════════════════════════════════════════

def draw_result(annotated, best, a_depth, b_depth, target_depth, iteration, delta_mm, move_m):
    """在标注图上绘制 a/b 连线、中点、顶部数据栏"""
    img = annotated.copy()
    h, w = img.shape[:2]

    a_cx, a_cy = int(best["a"]["cx"]), int(best["a"]["cy"])
    b_cx, b_cy = int(best["b"]["cx"]), int(best["b"]["cy"])

    # a/b 连线
    cv2.line(img, (a_cx, a_cy), (b_cx, b_cy), COLOR_LINE_AB, 2)
    # 中点
    mid_x = int((a_cx + b_cx) / 2)
    mid_y = int((a_cy + b_cy) / 2)
    cv2.circle(img, (mid_x, mid_y), 6, COLOR_MID, -1)

    # 顶部数据栏
    avg_depth = (a_depth + b_depth) / 2
    line1 = (f"iter:{iteration}  a:{a_depth:.0f}mm  b:{b_depth:.0f}mm  "
             f"avg:{avg_depth:.0f}mm  target:{target_depth}mm")
    line2 = (f"delta:{delta_mm:+.0f}mm  move:{move_m*1000:+.0f}mm")
    bar_h = 55
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    font_scale = 0.6
    thickness = 2
    cv2.putText(bar, line1, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BAR_TEXT, thickness)
    cv2.putText(bar, line2, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_BAR_TEXT, thickness)
    img = np.vstack([bar, img])

    return img


# ═══════════════════════════════════════════════════════════
#  纠偏主流程
# ═══════════════════════════════════════════════════════════

def run_correction(model, g2, target_depth, threshold, max_iter, output_dir, dry_run):
    """执行前后纠偏闭环

    返回: dict 纠偏结果
    """
    total_move_m = 0.0
    history = []
    # 安全检查: 记录上一次的移动方向和偏差
    last_move_dir = 0       # +1 前进, -1 后退, 0 无
    last_avg_depth = None   # 上一次的平均深度
    last_delta = None       # 上一次的偏差(用于安全检查)

    print(f"\n{'='*60}")
    print(f"开始底盘前后纠偏")
    print(f"  目标深度: {target_depth} mm")
    print(f"  阈值:     {threshold} mm")
    print(f"  最大迭代: {max_iter}")
    print(f"  单次最大移动: {MAX_SINGLE_MOVE*1000:.0f} mm")
    if dry_run:
        print(f"  *** DRY RUN: 只检测不移动 ***")
    print(f"{'='*60}")

    for iteration in range(1, max_iter + 1):
        print(f"\n--- 迭代 {iteration}/{max_iter} ---")

        # 1. 拍照 (彩色 + 深度), 最多重试3次
        color_img = None
        depth_2d = None
        for attempt in range(1, 4):
            print(f"[{iteration}] 拍照中... (第{attempt}次)")
            color_img, depth_2d = capture_color_and_depth()
            if color_img is not None and depth_2d is not None:
                break
            if attempt < 3:
                print(f"[{iteration}] 拍照失败, 等待重试...")
                time.sleep(1.0)

        if color_img is None or depth_2d is None:
            print(f"[{iteration}] ✗ 3次拍照均失败, 终止纠偏")
            return {"success": False, "reason": "拍照失败", "iteration": iteration, "history": history}

        # 2. YOLO 检测 a/b
        res = get_midpoint(model, color_img)
        if res is None:
            print(f"[{iteration}] ✗ 未检测到目标, 终止纠偏")
            cv2.imwrite(os.path.join(output_dir, f"fb_fail_iter{iteration}.jpg"), color_img)
            return {"success": False, "reason": "未检测到 a/b", "iteration": iteration, "history": history}

        mid_x, mid_y, best, annotated = res

        # 3. 获取 a/b 深度值
        a_cx, a_cy = int(best["a"]["cx"]), int(best["a"]["cy"])
        b_cx, b_cy = int(best["b"]["cx"]), int(best["b"]["cy"])
        a_depth = get_depth_at_point(depth_2d, a_cx, a_cy)
        b_depth = get_depth_at_point(depth_2d, b_cx, b_cy)
        avg_depth = (a_depth + b_depth) / 2

        if a_depth == 0 or b_depth == 0:
            print(f"[{iteration}] ✗ 深度值无效 (a={a_depth}, b={b_depth}), 终止纠偏")
            cv2.imwrite(os.path.join(output_dir, f"fb_fail_iter{iteration}.jpg"), annotated)
            return {"success": False, "reason": "深度值无效", "iteration": iteration, "history": history}

        # 4. 计算偏差
        delta_mm = avg_depth - target_depth
        print(f"[{iteration}] a={a_depth:.0f}mm  b={b_depth:.0f}mm  avg={avg_depth:.0f}mm  "
              f"target={target_depth}mm  delta={delta_mm:+.0f}mm")

        # 4.5 安全检查: 移动后偏差|delta|应减小, 若反而增大超过容差 → 终止
        #     场景: 后退后深度增大是正常的(delta从-65→-3), 不应触发
        #           但如果后退后深度反而减小(delta从-65→-120), 说明异常
        curr_move_dir = 1 if delta_mm > 0 else -1
        if last_move_dir != 0 and last_delta is not None:
            # 比较移动前后 |delta| 的变化
            delta_abs_change = abs(delta_mm) - abs(last_delta)
            if delta_abs_change > 30:
                # 偏差反而增大超过30mm, 说明移动方向错误或深度图不同步
                dir_name = "前进" if last_move_dir > 0 else "后退"
                print(f"[{iteration}] ⚠⚠ 安全终止: {dir_name}后偏差反而增大, "
                      f"|delta|从 {abs(last_delta):.0f}mm 增大到 {abs(delta_mm):.0f}mm "
                      f"(+{delta_abs_change:.0f}mm), 可能深度图与彩色图不同步")
                result_img = draw_result(annotated, best, a_depth, b_depth, target_depth,
                                         iteration, delta_mm, 0.0)
                fname = f"fb_safety_stop_iter{iteration}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(os.path.join(output_dir, fname), result_img)
                history.append((iteration, avg_depth, delta_mm, 0.0))
                return {"success": False, "reason": "安全终止: 偏差异常增大",
                        "iteration": iteration, "history": history,
                        "image": os.path.join(output_dir, fname)}

        # 5. 判断收敛
        if abs(delta_mm) < threshold:
            print(f"[{iteration}] ✓ 已收敛 (|{delta_mm:.0f}| < {threshold})")
            result_img = draw_result(annotated, best, a_depth, b_depth, target_depth,
                                     iteration, delta_mm, 0.0)
            fname = f"fb_final_iter{iteration}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), result_img)
            history.append((iteration, avg_depth, delta_mm, 0.0))
            return {"success": True, "converged": True, "iterations": iteration,
                    "final_depth": avg_depth, "final_delta": delta_mm,
                    "a_depth": a_depth, "b_depth": b_depth,
                    "total_move": total_move_m, "history": history,
                    "image": os.path.join(output_dir, fname)}

        # 6. 计算移动量
        # dx>0=前进(靠近,深度减小), dx<0=后退(远离,深度增大)
        # delta=avg-target: delta>0太远需前进, delta<0太近需后退
        # 之前符号错误导致后退撞墙,正确为正号(与lr纠偏不同)
        move_m = delta_mm / 1000.0
        # 限制单次最大移动
        if abs(move_m) > MAX_SINGLE_MOVE:
            move_m = MAX_SINGLE_MOVE if move_m > 0 else -MAX_SINGLE_MOVE
            print(f"[{iteration}] ⚠ 移动量过大, 限制为 {move_m*1000:.0f}mm")

        direction = "前进" if move_m > 0 else "后退"
        print(f"[{iteration}] 需移动: dx = {move_m*1000:+.0f}mm ({direction})")

        # 7. 保存纠偏前图像
        before_img = draw_result(annotated, best, a_depth, b_depth, target_depth,
                                 iteration, delta_mm, move_m)
        before_fname = f"fb_iter{iteration}_before_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(output_dir, before_fname), before_img)

        history.append((iteration, avg_depth, delta_mm, move_m))

        # 更新安全检查状态
        last_move_dir = curr_move_dir
        last_avg_depth = avg_depth
        last_delta = delta_mm

        if dry_run:
            print(f"[{iteration}] (DRY RUN 跳过移动)")
            continue

        # 8. 执行底盘移动
        ok = move_chassis_relative(g2, dx_m=move_m)
        if not ok:
            print(f"[{iteration}] ✗ 底盘移动失败, 终止纠偏")
            return {"success": False, "reason": "底盘移动失败", "iteration": iteration, "history": history}

        total_move_m += move_m
        print(f"[{iteration}] 等待 {SETTLE_TIME}s 稳定...")
        time.sleep(SETTLE_TIME)

    # 达到最大迭代, 拍最终验证照
    print(f"\n[最终] 拍照验证...")
    color_img, depth_2d = capture_color_and_depth()
    if color_img is not None and depth_2d is not None:
        res = get_midpoint(model, color_img)
        if res is not None:
            mid_x, mid_y, best, annotated = res
            a_cx, a_cy = int(best["a"]["cx"]), int(best["a"]["cy"])
            b_cx, b_cy = int(best["b"]["cx"]), int(best["b"]["cy"])
            a_depth = get_depth_at_point(depth_2d, a_cx, a_cy)
            b_depth = get_depth_at_point(depth_2d, b_cx, b_cy)
            avg_depth = (a_depth + b_depth) / 2
            delta_mm = avg_depth - target_depth
            print(f"[最终] a={a_depth:.0f}mm  b={b_depth:.0f}mm  avg={avg_depth:.0f}mm  delta={delta_mm:+.0f}mm")

            result_img = draw_result(annotated, best, a_depth, b_depth, target_depth,
                                     max_iter, delta_mm, 0.0)
            fname = f"fb_final_iter{max_iter}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(output_dir, fname), result_img)

            return {"success": True, "converged": abs(delta_mm) < threshold,
                    "iterations": max_iter, "final_depth": avg_depth,
                    "final_delta": delta_mm, "a_depth": a_depth, "b_depth": b_depth,
                    "total_move": total_move_m, "history": history,
                    "image": os.path.join(output_dir, fname)}

    return {"success": True, "converged": False, "iterations": max_iter,
            "total_move": total_move_m, "history": history}


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    global REMOTE_INFER, REMOTE_INFER_FALLBACK
    parser = argparse.ArgumentParser(description="底盘前后纠偏 (深度图)")
    parser.add_argument("--target-depth", type=int, default=DEFAULT_TARGET_DEPTH,
                        help=f"目标深度 (mm), 默认 {DEFAULT_TARGET_DEPTH}")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"收敛阈值 (mm), 默认 {DEFAULT_THRESHOLD}")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER,
                        help=f"最大迭代次数, 默认 {DEFAULT_MAX_ITER}")
    parser.add_argument("--dry-run", action="store_true",
                        help="只检测不移动")
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载模型 (远程推理时可选加载本地模型作回退)
    if REMOTE_INFER:
        if REMOTE_INFER_FALLBACK:
            print(f"[1] 远程推理模式 + 本地回退, 加载本地模型: {MODEL_PATH}")
            model = YOLO(MODEL_PATH)
        else:
            print(f"[1] 远程推理模式 (无回退), 跳过本地模型加载")
            model = None
    else:
        print(f"[1] 加载 YOLO 模型: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    print(f"    task=detect, names={model.names if model else '(远程模式)'}")

    g2 = None
    if not args.dry_run:
        print(f"[2] 连接机器人 (MQTT {MQTT_BROKER}:{MQTT_PORT})...")
        g2 = setup_minth()
        print(f"    ✓ Minth 已就绪")
    else:
        print(f"[2] DRY RUN 模式, 跳过机器人连接")

    result = run_correction(
        model=model,
        g2=g2,
        target_depth=args.target_depth,
        threshold=args.threshold,
        max_iter=args.max_iter,
        output_dir=OUTPUT_DIR,
        dry_run=args.dry_run,
    )

    # 打印总结
    print(f"\n{'='*60}")
    print(f"纠偏总结")
    print(f"{'='*60}")
    print(f"成功:     {result.get('success', False)}")
    print(f"收敛:     {result.get('converged', False)}")
    print(f"迭代次数: {result.get('iterations', 0)}")

    if "a_depth" in result:
        print(f"a 深度:   {result['a_depth']:.0f} mm")
        print(f"b 深度:   {result['b_depth']:.0f} mm")
        print(f"平均深度: {result.get('final_depth', 0):.0f} mm")
        print(f"最终偏差: {result.get('final_delta', 0):+.0f} mm")

    print(f"总移动量: {result.get('total_move', 0)*1000:+.0f} mm")

    if "history" in result and result["history"]:
        print(f"\n迭代历史:")
        print(f"  {'iter':>4}  {'avg_depth':>10}  {'delta_mm':>10}  {'move_mm':>10}")
        for iter_i, avg_d, delta_d, move_d in result["history"]:
            print(f"  {iter_i:>4}  {avg_d:>10.0f}  {delta_d:>+10.0f}  {move_d*1000:>+10.0f}")

    if "image" in result:
        print(f"\n最终图像: {result['image']}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
