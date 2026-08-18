#!/usr/bin/env python3
"""
头部 RGB 相机拍照 → YOLO 模型标注 → 几何处理（连线/中垂线/斜率）
一体化流程，最终输出与 results_line 文件夹内图像相同的效果。

流程:
  1. 通过 agibot_gdk 获取头部彩色相机图像
  2. 用 best_new.pt 模型检测 a/b 两个目标框
  3. 取 a/b 框中心点连线，标注斜率/角度/中点/中垂线/图像中心竖直线
  4. 保存结果到 output 目录

用法:
  python capture_and_annotate.py                          # 拍照+标注，结果存到 output/
  python capture_and_annotate.py --model best_new.pt      # 指定模型
  python capture_and_annotate.py --out output             # 指定输出目录
  python capture_and_annotate.py --no-camera --src yolo_images_all  # 不用相机，改用本地图片
"""
import argparse
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== 默认配置 =====================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(BASE_DIR / "best_new.pt")
DEFAULT_OUT = str(BASE_DIR / "output")


# ═══════════════════════════════════════════════════════════
#  相机拍照
# ═══════════════════════════════════════════════════════════

def capture_head_color(broker="localhost", port=1883, timeout=10.0):
    """通过 MQTT 订阅 /humanoid/camera/data 获取头部彩色相机图像

    流程:
      1. 连接 MQTT broker
      2. 发布 {"command":"start"} 到 /humanoid/camera/control 启动相机流
      3. 订阅 /humanoid/camera/data 等待一帧
      4. 解析 head_color (base64 JPEG) → numpy BGR 数组
      5. 发布 {"command":"stop"} 停止相机流

    Parameters
    ----------
    broker : str
        MQTT broker 地址 (机器人本体为 localhost)
    port : int
        MQTT 端口
    timeout : float
        等待相机帧的最长时间（秒）
    """
    import base64 as _b64
    import json as _json
    import paho.mqtt.client as mqtt

    TOPIC_DATA = "/humanoid/camera/data"
    TOPIC_CTRL = "/humanoid/camera/control"

    received = {"img": None}

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[MQTT] 已连接 {broker}:{port}")
            client.subscribe(TOPIC_DATA, qos=0)
            # 启动相机流发布
            client.publish(TOPIC_CTRL, _json.dumps({"command": "start"}), qos=0)
            print(f"[MQTT] 已订阅 {TOPIC_DATA} 并发送 start 命令")
        else:
            print(f"[MQTT] 连接失败 rc={rc}")

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
                    print(f"[MQTT] 收到 head_color 帧: {bgr.shape[1]}x{bgr.shape[0]}")
                    client.disconnect()
        except Exception as e:
            print(f"[MQTT] 解析失败: {e}")

    print("[相机] 通过 MQTT 获取头部彩色图像...")
    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port, keepalive=60)

    t_start = time.time()
    while received["img"] is None and time.time() - t_start < timeout:
        client.loop(timeout=0.2)

    # 停止相机流
    try:
        client.publish(TOPIC_CTRL, _json.dumps({"command": "stop"}), qos=0)
    except Exception:
        pass
    try:
        client.disconnect()
    except Exception:
        pass

    if received["img"] is None:
        raise RuntimeError(f"在 {timeout}s 内未收到相机数据，请确认机器人相机服务在运行")

    print(f"[相机] 图像尺寸: {received['img'].shape[1]}x{received['img'].shape[0]}")
    return received["img"]


# ═══════════════════════════════════════════════════════════
#  YOLO 模型标注 (参照 predict_batch.py)
# ═══════════════════════════════════════════════════════════

def run_yolo_annotation(model, img_bgr, imgsz=640, conf=0.25):
    """对图像执行 YOLO 推理，返回 (annotated_img, best_boxes_dict)

    annotated_img: 模型标注后的图像 (带检测框)
    best_boxes: {"a": {x1,y1,x2,y2,cx,cy,conf}, "b": {...}} 或 None
    """
    results = model(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r0 = results[0]

    # 模型自带绘制 (detect 画框+标签)
    annotated = r0.plot()
    boxes = r0.boxes
    names = model.names

    # 按类别收集置信度最高的框
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

    return annotated, best


# ═══════════════════════════════════════════════════════════
#  几何处理 (参照 draw_line_slope.py)
# ═══════════════════════════════════════════════════════════

def draw_geometry(img, best):
    """在图像上绘制几何标注，返回处理后的图像

    绘制内容:
      - a/b 检测框 (红/绿)
      - a/b 中心点 (绿色)
      - 中心连线 (黄色)
      - 线段中点 (橙色)
      - 经过中点的中垂线 (蓝色)
      - 图像中心点 (洋红) + 竖直线 (青色)
      - 顶部数据栏 (斜率/角度/中点坐标)
    """
    pt_a = best.get("a")
    pt_b = best.get("b")

    h, w = img.shape[:2]

    if not pt_a or not pt_b:
        # 未同时检测到 a/b，只画顶部提示栏
        bar_h = 30
        bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
        cv2.putText(bar, "WARNING: a or b not detected", (12, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return np.vstack([bar, img])

    cx1, cy1 = pt_a["cx"], pt_a["cy"]
    cx2, cy2 = pt_b["cx"], pt_b["cy"]

    # ====== 几何计算 ======
    dx = cx2 - cx1
    dy = cy2 - cy1
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)

    if abs(dx) < 1e-6:
        slope = float("inf")
        slope_str = "inf"
    else:
        slope = dy / dx
        slope_str = f"{slope:.4f}"

    mid_x_f = (cx1 + cx2) / 2
    mid_y_f = (cy1 + cy2) / 2
    mid_x = int(mid_x_f)
    mid_y = int(mid_y_f)

    img_cx = w / 2
    img_cy = h / 2

    # ====== 顶部数据栏 ======
    line1 = f"slope:{slope_str}  angle:{angle_deg:.1f}deg"
    line2 = f"mid:({mid_x},{mid_y})  img_mid:({int(img_cx)},{int(img_cy)})"
    bar_h = 55
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)
    font_scale = 0.6
    thickness = 2
    cv2.putText(bar, line1, (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    cv2.putText(bar, line2, (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    img = np.vstack([bar, img])
    offset_y = bar_h

    def off(pt):
        return (int(pt[0]), int(pt[1] + offset_y))

    # —— a/b 检测框 ——
    cv2.rectangle(img, (int(pt_a["x1"]), int(pt_a["y1"]) + offset_y),
                  (int(pt_a["x2"]), int(pt_a["y2"]) + offset_y), (0, 0, 255), 2)
    cv2.rectangle(img, (int(pt_b["x1"]), int(pt_b["y1"]) + offset_y),
                  (int(pt_b["x2"]), int(pt_b["y2"]) + offset_y), (0, 255, 0), 2)

    # —— 两个中心点 ——
    cv2.circle(img, off((cx1, cy1)), 6, (0, 255, 0), -1)
    cv2.circle(img, off((cx2, cy2)), 6, (0, 255, 0), -1)

    # —— 中心连线 (黄色) ——
    cv2.line(img, off((cx1, cy1)), off((cx2, cy2)), (0, 255, 255), 2)

    # —— 线段中点 (橙色) ——
    cv2.circle(img, off((mid_x_f, mid_y_f)), 6, (0, 165, 255), -1)

    # —— 经过中点的中垂线 (蓝色) ——
    LEN = max(w, h)
    perp_dx = -dy
    perp_dy = dx
    norm = math.hypot(perp_dx, perp_dy)
    if norm > 1e-6:
        perp_dx /= norm
        perp_dy /= norm
        p1 = (int(mid_x_f + perp_dx * LEN), int(mid_y_f + perp_dy * LEN + offset_y))
        p2 = (int(mid_x_f - perp_dx * LEN), int(mid_y_f - perp_dy * LEN + offset_y))
        cv2.line(img, p1, p2, (255, 0, 0), 2)

    # —— 图像中心点 (洋红) ——
    img_cx_i, img_cy_i = int(img_cx), int(img_cy + offset_y)
    cv2.circle(img, (img_cx_i, img_cy_i), 8, (255, 0, 255), -1)
    cv2.putText(img, "img_mid", (img_cx_i + 12, img_cy_i + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # —— 经过图像中心的竖直线 (青色) ——
    cv2.line(img, (img_cx_i, offset_y), (img_cx_i, h + offset_y), (255, 255, 0), 2)

    return img


# ═══════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="头部RGB拍照 → YOLO标注 → 几何处理")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 .pt 路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"输出目录 (默认: {DEFAULT_OUT})")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--no-camera", action="store_true", help="不使用相机，改用本地图片")
    parser.add_argument("--src", default=None, help="本地图片路径 (--no-camera 时使用)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # ====== Step 1: 获取图像 ======
    if args.no_camera:
        # 本地图片模式
        src = Path(args.src) if args.src else (BASE_DIR / "yolo_images_all")
        images = sorted([p for p in src.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        if not images:
            raise FileNotFoundError(f"未在 {src} 中找到图像")
        print(f"[1] 本地图片模式: {len(images)} 张")
    else:
        # 相机拍照模式
        print("[1] 头部相机拍照")
        images = None  # 稍后在循环中处理

    # ====== Step 2: 加载模型 ======
    print(f"[2] 加载模型: {args.model}")
    model = YOLO(args.model)
    print(f"    task={model.task}, names={model.names}")

    # ====== Step 3: 逐张处理 ======
    if args.no_camera:
        # 本地图片批量处理
        for i, img_path in enumerate(images, 1):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [{i}] 读取失败: {img_path.name}")
                continue
            _process_one(model, img_bgr, img_path.name, out_dir, args.imgsz, args.conf)
    else:
        # 相机单张拍照
        img_bgr = capture_head_color()
        fname = f"capture_{timestamp}.jpg"

        # 保存原图
        raw_path = out_dir / f"raw_{fname}"
        cv2.imwrite(str(raw_path), img_bgr)
        print(f"  原图已保存: {raw_path}")

        _process_one(model, img_bgr, fname, out_dir, args.imgsz, args.conf)

    print(f"\n结果目录: {out_dir}")


def _process_one(model, img_bgr, name, out_dir, imgsz, conf):
    """对单张图像执行: YOLO标注 → 几何处理 → 保存"""
    # Step A: YOLO 模型标注
    annotated, best = run_yolo_annotation(model, img_bgr, imgsz, conf)
    ann_path = out_dir / f"annotated_{name}"
    cv2.imwrite(str(ann_path), annotated)

    n_boxes = len(best)
    print(f"  YOLO标注: {ann_path.name}  ({n_boxes} 个目标: {list(best.keys())})")

    # Step B: 几何处理 (连线/中垂线/斜率)
    final_img = draw_geometry(annotated, best)
    final_path = out_dir / f"result_{name}"
    cv2.imwrite(str(final_path), final_img)

    if best.get("a") and best.get("b"):
        pt_a, pt_b = best["a"], best["b"]
        dx = pt_b["cx"] - pt_a["cx"]
        dy = pt_b["cy"] - pt_a["cy"]
        slope = "inf" if abs(dx) < 1e-6 else f"{dy/dx:.4f}"
        angle = math.degrees(math.atan2(dy, dx))
        print(f"  几何处理: {final_path.name}  slope={slope} angle={angle:.2f}deg")
    else:
        print(f"  几何处理: {final_path.name}  (未检测到 a/b)")


if __name__ == "__main__":
    main()
