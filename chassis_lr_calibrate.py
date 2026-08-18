#!/usr/bin/env python3
"""
底盘横移标定脚本

通过让底盘按预设距离序列左右横移，每次拍照并用 YOLO 检测 a/b 中心，
记录连线中点 x 像素变化，拟合 "像素 ↔ 米" 线性关系，输出标定系数。

流程:
  1. 在起点位置拍照 → 记录中点 x₀
  2. 底盘按序列横移 → 每个位置拍照 → 记录中点 xᵢ
  3. 回到起点
  4. 最小二乘拟合 Δpx = k * Δy_m + b
  5. 输出 CSV 和拟合结果

用法:
  python chassis_lr_calibrate.py
  python chassis_lr_calibrate.py --model /data/wzd/best_new.pt
  python chassis_lr_calibrate.py --output /data/wzd/calibration
  python chassis_lr_calibrate.py --offsets -0.2 -0.1 0 0.1 0.2
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== 路径配置 (可后续修改) =====================
# 模型路径
MODEL_PATH = "/data/wzd/best_new.pt"
# 输出目录
OUTPUT_DIR = "/data/wzd/calibration"
# GDK 服务目录 (用于导入 agibot_gdk)
GDK_SERVICES_DIR = "/data/wxf/wxf0721/services"
# Minth 控制库路径 (用于调用机器人底盘横移)
MINTH_DIR = "/data/wxf/wxf0721/runtime"

# ===================== 默认标定参数 =====================
# 横移偏移序列 (米), 正=左, 负=右 (与 go_rel 的 y 一致)
DEFAULT_OFFSETS = [-0.2, -0.1, 0.0, 0.1, 0.2]
# 每次移动后等待稳定时间 (秒)
SETTLE_TIME = 1.5
# MQTT 配置 (仅用于 minth 底盘控制)
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# GDK 相机对象 (延迟初始化)
_gdk_camera = None


# ═══════════════════════════════════════════════════════════
#  通过 GDK 直接拍照头部彩色相机 (不依赖 MQTT 相机流)
# ═══════════════════════════════════════════════════════════

def _init_gdk():
    """初始化 GDK 相机接口, 与 services/common.py 的 init_gdk() 一致"""
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
#  YOLO 推理: 获取 a/b 中心点
# ═══════════════════════════════════════════════════════════

def get_midpoint(model, img_bgr, imgsz=640, conf=0.25):
    """对图像执行 YOLO 推理，返回 (mid_x, mid_y, best_dict) 或 None"""
    results = model(img_bgr, imgsz=imgsz, conf=conf, verbose=False)
    r0 = results[0]
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

    if "a" not in best or "b" not in best:
        return None

    mid_x = (best["a"]["cx"] + best["b"]["cx"]) / 2
    mid_y = (best["a"]["cy"] + best["b"]["cy"]) / 2
    return mid_x, mid_y, best


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
#  标定主流程
# ═══════════════════════════════════════════════════════════

def run_calibration(model, g2, offsets, output_dir, imgsz, conf):
    """执行标定流程

    流程:
      1. 在起点拍照, 记录基准中点 x₀
      2. 按序列横移, 每个位置拍照记录中点 xᵢ
      3. 回到起点 (反向累计运动)
      4. 拟合 Δpx = k * Δy_m + b
    """
    records = []  # [(offset_m, mid_x, mid_y, img_path), ...]
    cumulative_y = 0.0  # 累计横移量, 用于回退

    print(f"\n{'='*60}")
    print(f"开始标定: {len(offsets)} 个点位")
    print(f"偏移序列 (米): {offsets}")
    print(f"{'='*60}")

    # 起点拍照
    print(f"\n[起点] 拍照中...")
    img0 = capture_head_color()
    res0 = get_midpoint(model, img0, imgsz, conf)
    if res0 is None:
        print(f"[起点] ✗ 未检测到 a/b, 终止标定")
        return None

    mid_x0, mid_y0, best0 = res0
    print(f"[起点] 中点=({mid_x0:.1f}, {mid_y0:.1f})  a.conf={best0['a']['conf']:.2f}  b.conf={best0['b']['conf']:.2f}")

    # 保存起点图
    raw_path = os.path.join(output_dir, "calib_0_origin.jpg")
    cv2.imwrite(raw_path, img0)
    annotated = model(img0, imgsz=imgsz, conf=conf, verbose=False)[0].plot()
    ann_path = os.path.join(output_dir, "calib_0_origin_annotated.jpg")
    cv2.imwrite(ann_path, annotated)

    records.append((0.0, mid_x0, mid_y0, raw_path))

    # 按序列横移 (每次从原点出发, 避免增量误差累积)
    for i, offset in enumerate(offsets, 1):
        if offset == 0.0:
            # 0 偏移直接复用起点数据
            records.append((0.0, mid_x0, mid_y0, raw_path))
            print(f"\n[{i}/{len(offsets)}] offset=0.0m  (复用起点)")
            continue

        print(f"\n[{i}/{len(offsets)}] 目标: 横移 {offset:+.3f} m ...")

        # 如果当前不在原点, 先回原点
        if abs(cumulative_y) > 1e-6:
            print(f"[{i}] 回原点 (反向 {-cumulative_y:+.3f} m)...")
            move_chassis_relative(g2, dy_m=-cumulative_y)
            time.sleep(SETTLE_TIME)
            cumulative_y = 0.0

        # 从原点移动到目标位置
        print(f"[{i}] 从原点移动到 {offset:+.3f} m...")
        ok = move_chassis_relative(g2, dy_m=offset)
        if not ok:
            print(f"[{i}] ✗ 横移失败, 跳过")
            continue
        cumulative_y = offset
        time.sleep(SETTLE_TIME)

        # 拍照
        print(f"[{i}] 拍照中...")
        img = capture_head_color()
        res = get_midpoint(model, img, imgsz, conf)
        if res is None:
            print(f"[{i}] ✗ 未检测到 a/b, 跳过")
            cv2.imwrite(os.path.join(output_dir, f"calib_{i}_fail_offset{offset:+.3f}.jpg"), img)
            continue

        mid_x, mid_y, best = res
        print(f"[{i}] offset={offset:+.3f}m  中点=({mid_x:.1f}, {mid_y:.1f})  "
              f"a.conf={best['a']['conf']:.2f}  b.conf={best['b']['conf']:.2f}")

        # 保存图像
        raw_path_i = os.path.join(output_dir, f"calib_{i}_offset{offset:+.3f}.jpg")
        cv2.imwrite(raw_path_i, img)
        annotated = model(img, imgsz=imgsz, conf=conf, verbose=False)[0].plot()
        ann_path = os.path.join(output_dir, f"calib_{i}_offset{offset:+.3f}_annotated.jpg")
        cv2.imwrite(ann_path, annotated)

        records.append((offset, mid_x, mid_y, raw_path_i))

    # 回到起点
    if cumulative_y != 0.0:
        print(f"\n[回退] 反向移动 {cumulative_y:.3f} m 回到起点...")
        move_chassis_relative(g2, dy_m=-cumulative_y)
        time.sleep(SETTLE_TIME)
        print("[回退] 完成")

    return records


# ═══════════════════════════════════════════════════════════
#  数据分析与拟合
# ═══════════════════════════════════════════════════════════

def analyze_and_save(records, output_dir):
    """拟合 Δpx = k * Δy_m + b, 保存 CSV 和结果"""
    if len(records) < 2:
        print(f"\n[分析] 有效数据点不足 ({len(records)}/2), 无法拟合")
        return None

    # 基准点 (offset=0)
    base = next((r for r in records if abs(r[0]) < 1e-6), records[0])
    base_x = base[1]

    # 计算 Δ
    offsets_m = []
    delta_px = []
    for offset_m, mid_x, mid_y, _ in records:
        offsets_m.append(offset_m)
        delta_px.append(mid_x - base_x)

    # 最小二乘拟合: Δpx = k * Δy_m + b
    x = np.array(offsets_m)
    y = np.array(delta_px)
    k, b = np.polyfit(x, y, 1)
    # 残差
    y_pred = k * x + b
    residuals = y - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    # R²
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # 像素 → 米 系数 (k 的倒数, 因为 Δpx = k*Δy → Δy = Δpx/k)
    px_to_meter = 1.0 / k if abs(k) > 1e-9 else float("inf")

    print(f"\n{'='*60}")
    print(f"标定结果")
    print(f"{'='*60}")
    print(f"数据点数: {len(records)}")
    print(f"基准点 (offset=0): mid_x = {base_x:.2f} px")
    print(f"")
    print(f"线性拟合: Δpx = {k:.2f} * Δy_m + ({b:.2f})")
    print(f"  斜率 k = {k:.2f} px/m   (底盘左移 1m → 中点 x 增加 {k:.2f} px)")
    print(f"  截距 b = {b:.2f} px")
    print(f"  RMSE   = {rmse:.2f} px")
    print(f"  R²     = {r2:.4f}")
    print(f"")
    print(f"像素 → 米 转换系数 (1/k):")
    print(f"  PX_TO_METER = {px_to_meter:.6f} m/px")
    print(f"  即 1 像素 ≈ {abs(px_to_meter)*1000:.3f} 毫米")
    print(f"  注: 正负号表示方向, 实际使用时根据 go_rel 的 y 方向定义取负号")
    print(f"{'='*60}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "calibration_data.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "offset_m", "mid_x_px", "mid_y_px", "delta_px_from_base", "image_path"])
        for i, (offset_m, mid_x, mid_y, img_path) in enumerate(records):
            writer.writerow([i, f"{offset_m:.4f}", f"{mid_x:.2f}", f"{mid_y:.2f}",
                             f"{mid_x - base_x:.2f}", img_path])
    print(f"\nCSV 已保存: {csv_path}")

    # 保存拟合结果 JSON
    result = {
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "num_points": len(records),
        "base_mid_x_px": base_x,
        "fit": {
            "slope_k_px_per_m": float(k),
            "intercept_b_px": float(b),
            "rmse_px": float(rmse),
            "r_squared": float(r2),
        },
        "px_to_meter": float(px_to_meter),
        "note": "正 k 表示底盘左移 (正 y) 时中点 x 增大; 实际纠偏时 y_meters = -delta_px * px_to_meter",
    }
    json_path = os.path.join(output_dir, "calibration_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"结果 JSON 已保存: {json_path}")

    # 保存拟合图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, color="blue", s=60, label="实测数据", zorder=3)
        x_fit = np.linspace(min(x) - 0.02, max(x) + 0.02, 100)
        y_fit = k * x_fit + b
        ax.plot(x_fit, y_fit, "r-", label=f"拟合: Δpx = {k:.2f}·Δy + ({b:.2f})\nR²={r2:.4f}")
        ax.set_xlabel("底盘横移 Δy (m, 正=左)")
        ax.set_ylabel("中点 x 像素变化 Δpx (px)")
        ax.set_title("底盘横移标定: 像素 vs 米")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig_path = os.path.join(output_dir, "calibration_fit.png")
        fig.savefig(fig_path, dpi=100)
        print(f"拟合图已保存: {fig_path}")
    except Exception as e:
        print(f"(拟合图保存失败: {e})")

    return result


# ═══════════════════════════════════════════════════════════
#  主函数
# ═══════════════════════════════════════════════════════════

def main():
    global MINTH_DIR, MQTT_BROKER, MQTT_PORT, SETTLE_TIME

    parser = argparse.ArgumentParser(description="底盘横移标定: 像素 ↔ 米")
    parser.add_argument("--model", default=MODEL_PATH, help=f"YOLO 模型路径 (默认: {MODEL_PATH})")
    parser.add_argument("--output", default=OUTPUT_DIR, help=f"输出目录 (默认: {OUTPUT_DIR})")
    parser.add_argument("--minth-dir", default=MINTH_DIR, help=f"minth 库路径 (默认: {MINTH_DIR})")
    parser.add_argument("--offsets", type=float, nargs="+", default=DEFAULT_OFFSETS,
                        help=f"横移偏移序列(米), 正=左 负=右 (默认: {DEFAULT_OFFSETS})")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 推理尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--settle", type=float, default=SETTLE_TIME, help=f"移动后稳定时间(秒) (默认: {SETTLE_TIME})")
    parser.add_argument("--broker", default=MQTT_BROKER, help=f"MQTT broker (默认: {MQTT_BROKER})")
    parser.add_argument("--port", type=int, default=MQTT_PORT, help=f"MQTT 端口 (默认: {MQTT_PORT})")
    args = parser.parse_args()

    # 更新全局变量
    MINTH_DIR = args.minth_dir
    MQTT_BROKER = args.broker
    MQTT_PORT = args.port
    SETTLE_TIME = args.settle

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 检查模型
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    # 加载模型
    print(f"[1] 加载 YOLO 模型: {args.model}")
    model = YOLO(args.model)
    print(f"    task={model.task}, names={model.names}")

    # 初始化机器人控制
    print(f"[2] 连接机器人 (MQTT {MQTT_BROKER}:{MQTT_PORT})...")
    g2 = setup_minth()
    print(f"    ✓ Minth 已就绪")

    # 偏移序列去重并排序 (保留 0 作为基准)
    offsets = sorted(set(args.offsets))
    print(f"[3] 标定偏移序列: {offsets} m")

    # 执行标定
    records = run_calibration(model, g2, offsets, args.output, args.imgsz, args.conf)

    # 释放机器人连接
    g2.close()

    # 分析并保存结果
    if records:
        analyze_and_save(records, args.output)

    print(f"\n标定完成. 所有文件保存在: {args.output}")


if __name__ == "__main__":
    main()
