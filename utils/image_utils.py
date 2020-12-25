from IPython import display
from PIL import Image

import numpy as np
import cv2

from utils.plots import color_list


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a : np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
        (N, M, 3) is an NxM BGR color image. shape (N, M, 4) is an NxM BGRA color
        image.
    """
    a = a.clip(0, 255).astype('uint8')
    # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
            a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
            a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    display.display(Image.fromarray(a))


def box_cxcywh_to_xyxy(bbox):
    y = np.zeros_like(bbox)
    y[:, 0] = bbox[:, 0] - 0.5 * bbox[:, 2]  # top left x
    y[:, 1] = bbox[:, 1] - 0.5 * bbox[:, 3]  # top left y
    y[:, 2] = bbox[:, 0] + 0.5 * bbox[:, 2]  # bottom right x
    y[:, 3] = bbox[:, 1] + 0.5 * bbox[:, 3]  # bottom right y
    return y


def decode_image(images):
    images = images.permute(0, 2, 3, 1)
    images = to_numpy(images)
    images = images * 255
    return images


def decode_target(targets, sizes):
    h, w = sizes
    gain = np.ones(6)
    gain[2:6] = np.array([w, h, w, h])
    targets = targets * gain
    return targets


def bbox_visualize(imgs, targets):
    targets = to_numpy(targets)
    h, w = imgs.shape[-2:]
    imgs = decode_image(imgs)
    targets = decode_target(targets, (h, w))

    for i in range(imgs.shape[0]):
        img = imgs[i].astype(np.uint8)
        img = img[..., ::-1]
        target = targets[targets[:, 0] == i][:, 2:]
        target = box_cxcywh_to_xyxy(target)
        show_bbox(img, target)


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def anchor_match_visualize(images, targets, indices, anchors, pred):
    h, w = images.shape[-2:]
    images = decode_image(images)

    strdie = [8, 16, 32]

    images_with_anchor = []
    # 对每张图片进行可视化
    for j in range(images.shape[0]):
        img = images[j].astype(np.uint8)[..., ::-1]

        # 对每个预测尺度进行单独可视化
        vis_imgs = []
        for i in range(3):  # i=0 检测小物体，i=1 检测中等尺度物体，i=2 检测大物体
            s = strdie[i]
            # anchor尺度
            gain = np.array(pred[i].shape)[[3, 2, 3, 2]]
            b, _, grid_x, grid_y = indices[i]

            b = to_numpy(b)
            grid_x = to_numpy(grid_x)
            grid_y = to_numpy(grid_y)
            anchor = to_numpy(anchors[i])
            target = to_numpy(targets[i])

            # 找出对应图片对应分支的信息
            idx = b == j
            grid_x = grid_x[idx]
            grid_y = grid_y[idx]
            anchor = anchor[idx]
            target = target[idx]

            # 还原到原图尺度进行可视化
            target /= gain
            target *= np.array([w, h, w, h], np.float32)
            target = box_cxcywh_to_xyxy(target)

            # label 可视化
            img = show_bbox(img.copy(), target, color=(0, 0, 255), is_show=False)

            # anchor 需要考虑偏移，在任何一层，每个bbox最多3*3=9个anchor进行匹配
            anchor *= s
            anchor_bbox = np.stack([grid_y, grid_x], axis=1)
            k = np.array(pred[i].shape, np.float)[[3, 2]]
            anchor_bbox = anchor_bbox / k
            anchor_bbox *= np.array([w, h], np.float32)
            anchor_bbox = np.concatenate([anchor_bbox, anchor], axis=1)
            anchor_bbox = box_cxcywh_to_xyxy(anchor_bbox)

            # 正样本anchor可视化
            img = show_bbox(img, anchor_bbox, color=(0, 255, 255), is_show=False)
            vis_imgs.append(img)

        per_image_with_anchor = merge_images_with_boundary(vis_imgs)
        images_with_anchor.append(per_image_with_anchor)

    return images_with_anchor


def show_bbox(image, bboxs_list, color=None,
              thickness=1, font_scale=0.3, wait_time_ms=0, names=None,
              is_show=True, is_without_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.

    :param image: numpy.ndarray.
    :param bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
    :param color: tuple.
    :param thickness: int.
    :param fontScale: float.
    :param wait_time_ms: int
    :param names: string: window name
    :param is_show: bool: whether to display during middle process
    :return: numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_copy = image.copy()
    txt = ''
    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f}, id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)
        if color is None:
            colors = color_list()  # list of colors
            color = colors[np.random.randint(0, len(colors))]

        if not is_without_mask:
            image_copy = cv2.rectangle(
                image_copy,
                (bbox_f[0], bbox_f[1]),
                (bbox_f[2], bbox_f[3]),
                color,
                thickness,
            )
        else:
            mask = np.zeros_like(image_copy, np.uint8)
            mask1 = cv2.rectangle(
                mask,
                (bbox_f[0], bbox_f[1]),
                (bbox_f[2], bbox_f[3]),
                color,
                -1,
            )
            mask = np.zeros_like(image_copy, np.uint8)
            mask2 = cv2.rectangle(
                mask,
                (bbox_f[0], bbox_f[1]),
                (bbox_f[2], bbox_f[3]),
                color,
                thickness,
            )
            mask2 = cv2.addWeighted(mask1, 0.5, mask2, 8, 0.0)
            image_copy = cv2.addWeighted(image_copy, 1.0, mask2, 0.6, 0.0)
        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(
                image_copy,
                txt,
                (bbox_f[0], bbox_f[1] - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
    if is_show:
        merge_images_with_boundary(image_copy)
    return image_copy


def merge_images_with_boundary(imgs, row_col_num=(1, -1)):
    """
        Displays an image or a list of images in specified windows or self-initiated windows.
        You can also control display wait time by parameter 'wait_time_ms'.
        Additionally, this function provides an optional parameter 'is_merge' to
        decide whether to display all imgs in a particular window 'merge'.
        Besides, parameter 'row_col_num' supports user specified merge format.
        Notice, specified format must be greater than or equal to imgs number.

        :param imgs: numpy.ndarray or list.
        :param window_names: specified or None, if None, function will create different windows as '1', '2'.
        :param wait_time_ms: display wait time.
        :param is_merge: whether to merge all images.
        :param row_col_num: merge format. default is (1, -1), image will line up to show.
                            example=(2, 5), images will display in two rows and five columns.
        """
    if not isinstance(imgs, list):
        imgs = [imgs]

    img_merged = merge_images(imgs, row_col_num)
    return img_merged


def merge_images(imgs, row_col_num):
    """
        Merges all input images as an image with specified merge format.

        :param imgs : img list
        :param row_col_num : number of rows and columns displayed
        :return img : merges img
        """

    length = len(imgs)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'
    colors = color_list()  # list of colors
    color = colors[np.random.randint(0, len(colors))]

    for img in imgs:
        cv2.rectangle(img, (0, 0), (img.shape[1], img.shape[0]), color)

    if row_col_num[1] < 0 or length < row:
        img_merged = np.hstack(imgs)
    elif row_col_num[0] < 0 or length < col:
        img_merged = np.vstack(imgs)
    else:
        assert row * col >= length, 'Imgs overboundary, not enough windows to display all imgs!'

        fill_img_list = [np.zeros(imgs[0].shape, dtype=np.uint8)] * (row * col - length)
        imgs.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(imgs[start: end])
            merge_imgs_col.append(merge_col)

        img_merged = np.vstack(merge_imgs_col)

    return img_merged
