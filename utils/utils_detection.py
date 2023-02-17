import yaml
import json
import torch
import cv2 as cv
import numpy as np
import torchvision


def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)


def json_load(file='data.json'):
    with open(file, "r") as f:
        return json.load(f)


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y


def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n,2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = w * x[:, 0] + padw  # top left x
    y[:, 1] = h * x[:, 1] + padh  # top left y
    return y


def letterbox_image(image, return_padding=False):
    """
        为保持h,w的一致,对图片短边两侧进行等距离padding
    """
    h, w = image.shape[:2]

    if h > w:
        p = int((h - w) // 2)
        image = cv.copyMakeBorder(image, 0, 0, p, (h - w - p), cv.BORDER_CONSTANT, value=0)
    else:
        p = int((w - h) // 2)
        image = cv.copyMakeBorder(image, p, (w - h - p), 0, 0, cv.BORDER_CONSTANT, value=0)

    if return_padding:
        return image, p
    else:
        return image

def image_trans(img, size):
    scale = min((size[0] / img.shape[0]), (size[1] / img.shape[1]), 1.1)
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img_new = cv.resize(img, new_size, interpolation=cv.INTER_LINEAR)
    top = round((size[0] - new_size[1]) * 0.5)
    bottom = (size[0] - new_size[1]) - top
    left = round((size[1] - new_size[0]) * 0.5)
    right = (size[1] - new_size[0]) - left
    img_new = cv.copyMakeBorder(img_new, top, bottom, left, right, cv.BORDER_CONSTANT, value=0)
    img_new = img_new.transpose((2, 0, 1))[::-1]
    img_new = np.expand_dims(img_new, 0)
    img_new = np.ascontiguousarray(img_new).astype(np.float32)
    img_new = img_new / 255.0
    return img_new


def scale_bboxes(bboxes, img_ori_hw, img_det_hw):
    assert len(img_ori_hw) == len(img_ori_hw)

    scale = max(img_ori_hw[0] / img_det_hw[0], img_ori_hw[1] / img_det_hw[1])
    bboxes[:, :4] = bboxes[:, :4] * scale

    h_bias = (max(img_ori_hw) - img_ori_hw[0]) / 2.0
    w_bias = (max(img_ori_hw) - img_ori_hw[1]) / 2.0

    bboxes[:, [0, 2]] -= w_bias
    bboxes[:, [1, 3]] -= h_bias

    clip_boxes(bboxes, img_ori_hw)

    return bboxes


def scale_bboxes_v2(bboxes, img_ori_hw, img_det_hw, p):
    assert len(img_ori_hw) == len(img_ori_hw)

    scale = max(img_ori_hw[0] / img_det_hw[0], img_ori_hw[1] / img_det_hw[1])
    bboxes[:, :4] = bboxes[:, :4] * scale
    if img_ori_hw[0] > img_ori_hw[1]:
        bboxes[:, [0, 2]] -= p
    else:
        bboxes[:, [1, 3]] -= p

    clip_boxes(bboxes, img_ori_hw)

    return bboxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2



def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


def draw_boxes(img, boxes, scores, labels, catid_labels, textscale=1, color_dicts=None):
    boxes = tuple(boxes.astype('int'))
    if color_dicts is None:
        color_dicts = {k:(0,0,255) for k in labels.keys}

    text_size, _ = cv.getTextSize(f'{catid_labels[labels]}:{scores:.2f}', fontFace=cv.FONT_HERSHEY_DUPLEX,
                                  fontScale=textscale, thickness=1)
    text_w, text_h = text_size
    # fillarea = np.asarray([boxes[:2], [boxes[0] + text_w + 1, boxes[1]],
    #                        [boxes[0] + text_w + 1, boxes[1] + text_h + 2], [boxes[0], boxes[1] + text_h + 2]])
    img0 = cv.rectangle(img, boxes[:2], boxes[2:], thickness=2, lineType=cv.LINE_AA,
                        color=color_dicts[labels])
    # img0 = cv.fillPoly(img0, [fillarea], color=color_dicts[labels])
    img0 = cv.rectangle(img0, boxes[:2], (boxes[0] + text_w + 1, boxes[1] + text_h + 2),
                        thickness=-1, color=color_dicts[labels])
    img0 = cv.putText(img0, f'{catid_labels[labels]}:{scores:.2f}',
                      (boxes[0], boxes[1] + text_h),
                      fontFace=cv.FONT_HERSHEY_DUPLEX, fontScale=textscale, thickness=1,
                      lineType=cv.LINE_AA,
                      color=(255, 255, 255)
                      )
    return img0



def non_max_suppression(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        agnostic=False,
                        max_det=300):
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
    return output


def non_max_suppression_v2(prediction,
                        conf_thres=0.25,
                        iou_thres=0.45,
                        agnostic=False,
                        max_det=300):
    bs = prediction.shape[0]  # batch size
    xc = prediction[..., 4] > conf_thres  # candidates
    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS
    output = [np.zeros((0, 6), dtype=np.float32)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Detections matrix nx6 (xywh, conf, cls)
        j = x[:, 5:].argmax(axis=1, keepdims=True)
        # conf = x[:, 5:].max(axis=1, keepdims=True)
        conf = x[:, 5:]
        conf = conf[range(len(j)), j.ravel()].reshape(-1, 1)
        x = np.concatenate((x[:,:4], conf, j), 1)[conf.ravel() > conf_thres]
        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence

        # Batched NMS
        # c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        # i = cv.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
        c = x[:, 5].ravel().astype("int32")
        i = cv.dnn.NMSBoxesBatched(x[:, :4], x[:, 4], c, conf_thres, iou_thres, None, max_det)
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(x[:, :4][i], x[:, :4]) > iou_thres  # iou matrix
            weights = iou * x[:, 4][None]  # box weights
            x[i, :4] = np.matmul(weights, x[:, :4]) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = xywh2xyxy(x[i])
    return output


def yolox_postprocess(outputs, img_size, p6=False):

    grids = []
    expanded_strides = []

    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    return outputs


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self, id_and_obj):
        base_hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                     '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        n = len(id_and_obj) / len(base_hexs)
        if n > 1:
            n = int(n) + 1
            base_hexs *= n

        self.obj_id = tuple(id_and_obj.keys())
        self.hex = base_hexs[:len(self.obj_id)]
        self.id_and_hex = {k: v for k, v in zip(self.obj_id, self.hex)}

    def get_id_and_colors(self):
        id_and_colors = {k: self.hex2rgb(f'#{v}') for k, v in self.id_and_hex.items()}
        return id_and_colors

    def hex2rgb(self, h):  # rgb order
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


