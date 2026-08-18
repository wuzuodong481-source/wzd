#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO MQTT 推理服务器
====================
独立进程, 在 GPU 上加载 YOLO 模型, 通过 MQTT 提供推理服务。

工作流程:
  1. 订阅 /yolo/infer/request
  2. 收到 base64 图片 + 参数, GPU 推理
  3. 发布检测结果到 /yolo/infer/response (带 request_id 关联)

启动:
  python3 yolo_infer_server.py --model /data/wzd/best_new.pt --device 0

依赖:
  - ultralytics
  - paho-mqtt
  - opencv-python
  - numpy
"""
import os
import sys
import json
import time
import base64
import threading
import argparse
import uuid

import numpy as np
import cv2

# MQTT
import paho.mqtt.client as mqtt

# ultralytics (导入时尽量让 CUDA 初始化)
from ultralytics import YOLO


# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════
MQTT_BROKER_DEFAULT = "localhost"
MQTT_PORT_DEFAULT = 1883
TOPIC_REQUEST = "/yolo/infer/request"
TOPIC_RESPONSE = "/yolo/infer/response"
DEFAULT_IMGSZ = 640
DEFAULT_CONF = 0.25

JPEG_QUALITY = 85  # base64 图片传输质量


# ═══════════════════════════════════════════════════════════
#  推理服务器
# ═══════════════════════════════════════════════════════════
class YoloInferServer:
    def __init__(self, model_path, device="0", broker=MQTT_BROKER_DEFAULT, port=MQTT_PORT_DEFAULT):
        self.model_path = model_path
        self.device = device
        self.broker = broker
        self.port = port

        self.model = None
        self.client = None
        self._infer_lock = threading.Lock()
        self._infer_count = 0
        self._total_ms = 0.0
        self._running = False

    # ---------- 模型 ----------
    def load_model(self):
        print(f"[Server] 加载 YOLO 模型: {self.model_path}")
        print(f"[Server] 设备: {self.device}")
        # device="0" 表示使用 GPU
        self.model = YOLO(self.model_path)
        # 预热: 让 CUDA 完成初始化, 后续推理更快更稳
        print(f"[Server] 模型预热中...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.model(dummy, imgsz=DEFAULT_IMGSZ, conf=DEFAULT_CONF,
                       device=self.device, verbose=False)
        print(f"[Server] ✓ 模型加载并预热完成")

    # ---------- MQTT ----------
    def connect_mqtt(self):
        self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_start()
        print(f"[Server] MQTT 已连接 {self.broker}:{self.port}")

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        client.subscribe(TOPIC_REQUEST, qos=0)
        print(f"[Server] 订阅 {TOPIC_REQUEST}")

    def _on_message(self, client, userdata, msg):
        """收到推理请求 (异步处理, 不阻塞 MQTT loop)"""
        try:
            req = json.loads(msg.payload.decode("utf-8"))
        except Exception as e:
            print(f"[Server] 请求解析失败: {e}")
            return
        # 异步处理, 避免阻塞 MQTT 心跳
        t = threading.Thread(target=self._handle_request, args=(req,), daemon=True)
        t.start()

    def _handle_request(self, req):
        req_id = req.get("request_id", "")
        try:
            img_b64 = req.get("image")
            imgsz = int(req.get("imgsz", DEFAULT_IMGSZ))
            conf = float(req.get("conf", DEFAULT_CONF))

            # 解码图片
            buf = base64.b64decode(img_b64)
            arr = np.frombuffer(buf, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise ValueError("图片解码失败")

            # GPU 推理 (加锁避免并发)
            t0 = time.time()
            with self._infer_lock:
                results = self.model(img_bgr, imgsz=imgsz, conf=conf,
                                     device=self.device, verbose=False)
            elapsed_ms = (time.time() - t0) * 1000.0

            r0 = results[0]
            boxes = r0.boxes
            names = self.model.names

            # 构造检测结果 (与原 detect_ab 输出格式一致)
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

            # 标注图 (本地绘制, 不传回以节省带宽)
            # 客户端根据 boxes 自行绘制
            response = {
                "request_id": req_id,
                "success": True,
                "all_boxes": all_boxes,
                "best": best,
                "names": names,
                "elapsed_ms": elapsed_ms,
            }

            # 统计
            self._infer_count += 1
            self._total_ms += elapsed_ms
            avg_ms = self._total_ms / self._infer_count
            print(f"[Server] #{self._infer_count} 推理 {elapsed_ms:.1f}ms "
                  f"(avg {avg_ms:.1f}ms) 检测 {len(all_boxes)} 个目标")

        except Exception as e:
            import traceback
            traceback.print_exc()
            response = {
                "request_id": req_id,
                "success": False,
                "error": str(e),
            }

        # 发布响应
        try:
            self.client.publish(TOPIC_RESPONSE, json.dumps(response), qos=0)
        except Exception as e:
            print(f"[Server] 响应发布失败: {e}")

    # ---------- 运行 ----------
    def run(self):
        self.load_model()
        self.connect_mqtt()
        self._running = True
        print(f"[Server] 推理服务已启动, 等待请求...")
        print(f"[Server]   请求主题: {TOPIC_REQUEST}")
        print(f"[Server]   响应主题: {TOPIC_RESPONSE}")
        print(f"[Server] 按 Ctrl+C 退出")
        try:
            while self._running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print(f"\n[Server] 收到退出信号")
        finally:
            self.stop()

    def stop(self):
        self._running = False
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
        print(f"[Server] 已停止, 总推理 {self._infer_count} 次")


# ═══════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="YOLO MQTT 推理服务器")
    parser.add_argument("--model", default="/data/wzd/best_new.pt",
                        help="YOLO 模型路径")
    parser.add_argument("--device", default="0",
                        help="推理设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--broker", default=MQTT_BROKER_DEFAULT,
                        help="MQTT broker 地址")
    parser.add_argument("--port", type=int, default=MQTT_PORT_DEFAULT,
                        help="MQTT 端口")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"❌ 模型不存在: {args.model}")
        sys.exit(1)

    server = YoloInferServer(
        model_path=args.model,
        device=args.device,
        broker=args.broker,
        port=args.port,
    )
    server.run()


if __name__ == "__main__":
    main()
