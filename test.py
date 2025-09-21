# -*- coding: utf-8 -*-

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# -------------------------------
# 模型配置
# -------------------------------
RKNN_MODEL = "models/helmet_model.rknn"
MODEL_SIZE = (640, 640)
OBJ_THRESH = 0.55
NMS_THRESH = 0.45

# 你的类别
CLASSES = ["head", "helmet"]

# 初始化 RKNN
rknn_lite = RKNNLite()

def load_rknn_model():
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("❌ 加载 RKNN 模型失败！")
        exit(ret)
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("❌ 初始化 RKNN 运行时失败！")
        exit(ret)
    print("✅ RKNN 模型加载成功！")

# -------------------------------
# 工具函数
# -------------------------------
def letter_box(im, new_shape, pad_color=(0, 0, 0), info_need=True):
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    if info_need:
        return im, r, (dw, dh)
    return im

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.arange(mc, dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([MODEL_SIZE[1] // grid_w, MODEL_SIZE[0] // grid_h]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    keep = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    boxes = boxes[keep]
    classes = classes[keep]
    scores = (class_max_score * box_confidences)[keep]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    final_boxes, final_classes, final_scores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, s = boxes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            final_boxes.append(b[keep])
            final_classes.append(np.full(len(keep), c))
            final_scores.append(s[keep])
    if not final_boxes:
        return None, None, None
    return np.concatenate(final_boxes), np.concatenate(final_classes), np.concatenate(final_scores)

# -------------------------------
# 主流程
# -------------------------------
def run_inference(image_path, save_path="result.jpg"):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 无法读取图像: {image_path}")
        return

    orig_h, orig_w = frame.shape[:2]
    img, ratio, (dw, dh) = letter_box(frame, MODEL_SIZE)
    input_data = np.expand_dims(img, axis=0)

    # 模型推理
    outputs = rknn_lite.inference([input_data])
    boxes, classes, scores = post_process(outputs)

    if boxes is not None and len(classes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # === 坐标映射回原图 ===
            x1 = int((x1 - dw) / ratio)
            y1 = int((y1 - dh) / ratio)
            x2 = int((x2 - dw) / ratio)
            y2 = int((y2 - dh) / ratio)

            # 裁剪到原图边界
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            cls_id = int(classes[i])
            score = scores[i]
            label = f"{CLASSES[cls_id]} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"检测到 {len(classes)} 个目标")
    else:
        print("⚠️ 未检测到目标")

    cv2.imwrite(save_path, frame)
    print(f"结果已保存到 {save_path}")


# -------------------------------
if __name__ == '__main__':
    load_rknn_model()
    run_inference("/home/duzhong/test.jpg", "result.jpg")  # 输入图片路径 / 输出保存路径
    rknn_lite.release()
