#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量推理指定文件夹下所有图片, 标注结果保存到输出文件夹

使用 MQTT 远程 GPU 推理服务 (需先启动 yolo_infer_server.py)
"""
import os
import sys
import time
import glob
import cv2

# 复用 chassis_correct_all.py 的远程推理和绘制函数
sys.path.insert(0, "/data/wzd")
from chassis_correct_all import (
    _remote_infer,
    _draw_annotated,
    YOLO_IMGSZ,
    YOLO_CONF,
)


def batch_infer(input_dir, output_dir, imgsz=YOLO_IMGSZ, conf=YOLO_CONF):
    """批量推理 input_dir 下所有 jpg 图片, 标注图保存到 output_dir"""
    if not os.path.isdir(input_dir):
        print(f"[错误] 输入目录不存在: {input_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    # 收集所有 jpg (支持大小写)
    files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) +
                   glob.glob(os.path.join(input_dir, "*.jpeg")) +
                   glob.glob(os.path.join(input_dir, "*.png")))
    total = len(files)
    if total == 0:
        print(f"[错误] 输入目录无图片: {input_dir}")
        return False

    print(f"{'='*60}")
    print(f"批量推理")
    print(f"  输入: {input_dir}")
    print(f"  输出: {output_dir}")
    print(f"  图片数: {total}")
    print(f"  参数: imgsz={imgsz}, conf={conf}")
    print(f"{'='*60}")

    ok_cnt = 0
    fail_cnt = 0
    no_det_cnt = 0
    t_start = time.time()
    total_infer_ms = 0.0

    for i, fp in enumerate(files, 1):
        name = os.path.basename(fp)
        img = cv2.imread(fp)
        if img is None:
            print(f"[{i}/{total}] ✗ {name} (读取失败)")
            fail_cnt += 1
            continue

        t0 = time.time()
        resp = _remote_infer(img, imgsz=imgsz, conf=conf)
        dt = (time.time() - t0) * 1000

        if not resp.get("success"):
            print(f"[{i}/{total}] ✗ {name} (推理失败: {resp.get('error')}) [{dt:.0f}ms]")
            fail_cnt += 1
            continue

        total_infer_ms += resp.get("elapsed_ms", 0.0)
        all_boxes = resp.get("all_boxes", [])
        names = resp.get("names", {})
        annotated = _draw_annotated(img, all_boxes, names)

        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, annotated)

        n = len(all_boxes)
        if n == 0:
            no_det_cnt += 1
            tag = "(无检测)"
        else:
            ok_cnt += 1
            tag = f"({n} 目标)"

        print(f"[{i}/{total}] ✓ {name} {tag} [推理 {resp.get('elapsed_ms',0):.0f}ms / 往返 {dt:.0f}ms]")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"完成统计")
    print(f"  总数: {total}")
    print(f"  成功: {ok_cnt}")
    print(f"  无检测: {no_det_cnt}")
    print(f"  失败: {fail_cnt}")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  平均推理: {total_infer_ms/max(ok_cnt+no_det_cnt,1):.1f}ms/张")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="批量推理图片")
    p.add_argument("--input", "-i", default="/data/cg/cg/yolo_images_all",
                   help="输入图片目录")
    p.add_argument("--output", "-o", default="/data/wzd/jiazi",
                   help="输出标注图目录")
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    p.add_argument("--conf", type=float, default=YOLO_CONF)
    args = p.parse_args()

    ok = batch_infer(args.input, args.output, imgsz=args.imgsz, conf=args.conf)
    sys.exit(0 if ok else 1)
