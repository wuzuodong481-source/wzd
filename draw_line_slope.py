#!/usr/bin/env python3
"""
对前 10 张图像, 用模型检测出 a/b 两个框, 取各自中心点连线, 并标注斜率。
结果保存到 results_line 文件夹。

用法:
  python draw_line_slope.py
  python draw_line_slope.py --num 20          # 处理前 20 张
  python draw_line_slope.py --model best_new.pt --src yolo_images_all --out results_line
"""
import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ===================== 默认配置 =====================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(BASE_DIR / "best_new.pt")
DEFAULT_SRC = str(BASE_DIR / "yolo_images_all")
DEFAULT_OUT = str(BASE_DIR / "results_line")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    parser = argparse.ArgumentParser(description="检测 a/b 框中心点连线并标注斜率")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 .pt 路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"图像文件夹 (默认: {DEFAULT_SRC})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"结果输出文件夹 (默认: {DEFAULT_OUT})")
    parser.add_argument("--num", type=int, default=10, help="处理前 N 张图像 (默认: 10)")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    if not src_dir.exists():
        raise FileNotFoundError(f"图像文件夹不存在: {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"[1] 加载模型: {args.model}")
    model = YOLO(args.model)
    names = model.names
    print(f"    task={model.task}, names={names}")

    # 收集图像, 取前 N 张
    images = sorted([p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    images = images[:args.num]
    if not images:
        print(f"[!] 未在 {src_dir} 中找到图像")
        return
    print(f"[2] 处理前 {len(images)} 张图像 -> 输出到 {out_dir}")

    # 类名 -> class id 映射
    name_to_id = {v: k for k, v in names.items()}

    t_start = time.time()
    for i, img_path in enumerate(images, 1):
        # 推理
        results = model(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
        r0 = results[0]
        img = r0.orig_img.copy()
        boxes = r0.boxes

        # 按类别收集框 (取置信度最高的)
        best = {}
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = names.get(cls_id, str(cls_id))
            if cls_name not in best or conf > best[cls_name]["conf"]:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                best[cls_name] = {
                    "x1": float(x1), "y1": float(y1),
                    "x2": float(x2), "y2": float(y2),
                    "cx": float((x1 + x2) / 2),
                    "cy": float((y1 + y2) / 2),
                    "conf": conf,
                }

        pt_a = best.get("a")
        pt_b = best.get("b")

        if not pt_a or not pt_b:
            print(f"  [{i}/{len(images)}] {img_path.name} 跳过 (未同时检测到 a 和 b)")
            cv2.imwrite(str(out_dir / img_path.name), img)
            continue

        cx1, cy1 = pt_a["cx"], pt_a["cy"]
        cx2, cy2 = pt_b["cx"], pt_b["cy"]

        # ====== 计算几何数据 ======
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

        # 线段中点
        mid_x_f = (cx1 + cx2) / 2
        mid_y_f = (cy1 + cy2) / 2
        mid_x = int(mid_x_f)
        mid_y = int(mid_y_f)

        # 图像中心点
        h, w = img.shape[:2]
        img_cx = w / 2
        img_cy = h / 2

        # ====== 顶部数据栏 (黑底白字, 分两行显示) ======
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
        # 由于在顶部加了 bar, 所有 y 坐标下移 bar_h
        offset_y = bar_h

        def off(pt):
            return (int(pt[0]), int(pt[1] + offset_y))

        # —— 画 a/b 检测框 ——
        cv2.rectangle(img, (int(pt_a["x1"]), int(pt_a["y1"]) + offset_y),
                      (int(pt_a["x2"]), int(pt_a["y2"]) + offset_y), (0, 0, 255), 2)
        cv2.rectangle(img, (int(pt_b["x1"]), int(pt_b["y1"]) + offset_y),
                      (int(pt_b["x2"]), int(pt_b["y2"]) + offset_y), (0, 255, 0), 2)

        # —— 画两个中心点 (绿色实心圆) ——
        cv2.circle(img, off((cx1, cy1)), 6, (0, 255, 0), -1)
        cv2.circle(img, off((cx2, cy2)), 6, (0, 255, 0), -1)

        # —— a/b 中心连线 (黄色) ——
        cv2.line(img, off((cx1, cy1)), off((cx2, cy2)), (0, 255, 255), 2)

        # —— 线段中点 (橙色) ——
        cv2.circle(img, off((mid_x_f, mid_y_f)), 6, (0, 165, 255), -1)

        # —— 经过中点的中垂线 (蓝色) ——
        # 原线方向向量 (dx, dy), 垂线方向为 (-dy, dx) 或 (dy, -dx)
        LEN = max(w, h)  # 足够长以覆盖图像
        vx = dx
        vy = dy
        perp_dx = -vy
        perp_dy = vx
        norm = math.hypot(perp_dx, perp_dy)
        if norm > 1e-6:
            perp_dx /= norm
            perp_dy /= norm
            p1 = (int(mid_x_f + perp_dx * LEN), int(mid_y_f + perp_dy * LEN + offset_y))
            p2 = (int(mid_x_f - perp_dx * LEN), int(mid_y_f - perp_dy * LEN + offset_y))
            cv2.line(img, p1, p2, (255, 0, 0), 2)

        # —— 图像中心点 (粉色洋红) ——
        img_cx_i, img_cy_i = int(img_cx), int(img_cy + offset_y)
        cv2.circle(img, (img_cx_i, img_cy_i), 8, (255, 0, 255), -1)
        cv2.putText(img, "img_mid", (img_cx_i + 12, img_cy_i + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        # —— 经过图像中心点的竖直线 (青色) ——
        cv2.line(img, (img_cx_i, offset_y), (img_cx_i, h + offset_y), (255, 255, 0), 2)

        # 保存
        cv2.imwrite(str(out_dir / img_path.name), img)
        print(f"  [{i}/{len(images)}] {img_path.name}  slope={slope_str} angle={angle_deg:.2f}deg mid=({mid_x},{mid_y}) img_mid=({int(img_cx)},{int(img_cy)})")

    elapsed = time.time() - t_start
    print(f"\n完成: 处理 {len(images)} 张, 耗时 {elapsed:.1f}s")
    print(f"结果目录: {out_dir}")


if __name__ == "__main__":
    main()
