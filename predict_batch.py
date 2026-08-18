#!/usr/bin/env python3
"""
批量 YOLO 推理标注。
遍历 yolo_images_all 文件夹中的所有图像, 用 .pt 模型推理, 标注结果保存到 results 文件夹。
支持 detect 与 segment 任务 (自动判别)。

用法:
  python predict_batch.py                              # 用默认 best_new.pt 标注 yolo_images_all -> results
  python predict_batch.py --model other.pt             # 指定模型
  python predict_batch.py --src yolo_images_all --out results --model best_new.pt
"""
import argparse
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

# ===================== 默认配置 =====================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = str(BASE_DIR / "best_new.pt")
DEFAULT_SRC = str(BASE_DIR / "yolo_images_all")
DEFAULT_OUT = str(BASE_DIR / "results")
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    parser = argparse.ArgumentParser(description="批量 YOLO 推理标注")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型 .pt 路径 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--src", default=DEFAULT_SRC, help=f"图像文件夹 (默认: {DEFAULT_SRC})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"结果输出文件夹 (默认: {DEFAULT_OUT})")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸 (默认: 640)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    args = parser.parse_args()

    src_dir = Path(args.src)
    out_dir = Path(args.out)
    if not src_dir.exists():
        raise FileNotFoundError(f"图像文件夹不存在: {src_dir}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"模型文件不存在: {args.model}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"[1] 加载模型: {args.model}")
    model = YOLO(args.model)
    print(f"    task={model.task}, names={model.names}")

    # 收集图像
    images = sorted([p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not images:
        print(f"[!] 未在 {src_dir} 中找到图像 ({IMG_EXTS})")
        return
    print(f"[2] 找到 {len(images)} 张图像 -> 输出到 {out_dir}")

    # 批量推理
    n_ok = 0
    n_fail = 0
    t_start = time.time()
    for i, img_path in enumerate(images, 1):
        try:
            results = model(str(img_path), imgsz=args.imgsz, conf=args.conf, verbose=False)
            r0 = results[0]
            # ultralytics 自带绘制 (detect 画框, segment 画 mask 轮廓)
            annotated = r0.plot()
            # 保持相对子目录结构
            rel = img_path.relative_to(src_dir)
            dst = out_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), annotated)
            n_box = len(r0.boxes)
            print(f"  [{i}/{len(images)}] {rel.name} -> {rel}  ({n_box} 个目标)")
            n_ok += 1
        except Exception as e:
            print(f"  [{i}/{len(images)}] {img_path.name} 失败: {e}")
            n_fail += 1

    # 汇总
    elapsed = time.time() - t_start
    print("\n" + "=" * 50)
    print(f"完成: 成功 {n_ok}, 失败 {n_fail}, 耗时 {elapsed:.1f}s")
    print(f"结果目录: {out_dir}")


if __name__ == "__main__":
    main()
