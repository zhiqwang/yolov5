from IPython import display
from PIL import Image

import numpy as np
import cv2

from utils.plots import color_list


def cv2_imshow(a):
    """A replacement for cv2.imshow() for use in Jupyter notebooks.
    Args:
        a: np.ndarray. shape (N, M) or (N, M, 1) is an NxM grayscale image. shape
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


def to_numpy(tensor):
    if tensor.requires_grad:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.cpu().numpy()


def restore_label(target, feature_map_sizes, image_sizes):
    h, w = image_sizes
    target = target / np.array(feature_map_sizes)[[3, 2, 3, 2]]
    target = target * np.array([w, h, w, h], np.float32)
    target = box_cxcywh_to_xyxy(target)
    return target


def restore_anchor(anchor, grid_x, grid_y, stride, feature_map_size, image_sizes):
    anchor *= stride
    h, w = image_sizes
    anchor_restored = np.stack([grid_y, grid_x], axis=1)

    anchor_restored = anchor_restored / np.array(feature_map_size, np.float)[[3, 2]]
    anchor_restored = anchor_restored * np.array([w, h], np.float32)
    anchor_restored = np.concatenate([anchor_restored, anchor], axis=1)
    anchor_restored = box_cxcywh_to_xyxy(anchor_restored)

    return anchor_restored


def anchor_match_visualize(images, targets, indices, anchors, pred):
    image_sizes = images.shape[-2:]
    images = decode_image(images)

    strdie = [8, 16, 32]

    images_with_anchor = []
    # for loop all images
    for j in range(images.shape[0]):
        image = images[j].astype(np.uint8)[..., ::-1]

        # Visualize each prediction scale individually
        vis_all_scale_images = []
        # i = 0 is associate with small objects,
        # i = 1 is associate with intermediate objects,
        # i = 2 is associate with large objects
        for i in range(3):
            image_per_scale = image.copy()
            stride = strdie[i]
            # anchor scale
            b, _, grid_x, grid_y = indices[i]

            b, grid_x, grid_y, anchor, target = map(to_numpy, [b, grid_x, grid_y, anchors[i], targets[i]])

            # Find out the corresponding branch of one image
            idx = b == j
            grid_x, grid_y, anchor, target = grid_x[idx], grid_y[idx], anchor[idx], target[idx]

            # Restore to the original image scale for visualization
            target = restore_label(target, pred[i].shape, image_sizes)

            # visualize labels
            image_per_scale = overlay_bbox(
                image_per_scale,
                target,
                color=(0, 0, 255),
                thickness=3,
                with_mask=False,
            )

            # The anchors need to restore the offset.
            # In eacy layer there has at most 3x3=9 anchors for matching.
            anchor_restored = restore_anchor(anchor, grid_x, grid_y, stride, pred[i].shape, image_sizes)

            # visualize positive anchor
            image_per_scale = overlay_bbox(image_per_scale, anchor_restored)
            vis_all_scale_images.append(image_per_scale)

        per_image_with_anchor = merge_images_with_boundary(vis_all_scale_images)
        images_with_anchor.append(per_image_with_anchor)

    return images_with_anchor


def overlay_bbox(image, bboxs_list, color=None, thickness=2, font_scale=0.3, with_mask=False):
    """
    Visualize bbox in object detection by drawing rectangle.

    Args:
        image: numpy.ndarray.
        bboxs_list: list: [pts_xyxy, prob, id]: label or prediction.
        color: tuple.
        thickness: int.
        font_scale: float.
        with_mask: draw as mask.

    Return:
        numpy.ndarray
    """
    assert image is not None
    font = cv2.FONT_HERSHEY_SIMPLEX
    txt = ''
    colors = color_list()  # list of colors

    for bbox in bboxs_list:
        if len(bbox) == 5:
            txt = '{:.3f}'.format(bbox[4])
        elif len(bbox) == 6:
            txt = 'p={:.3f}, id={:.3f}'.format(bbox[4], bbox[5])
        bbox_f = np.array(bbox[:4], np.int32)

        mask = np.zeros_like(image, np.uint8)
        mask = cv2.rectangle(
            mask if with_mask else image,
            (bbox_f[0], bbox_f[1]),
            (bbox_f[2], bbox_f[3]),
            color=(color or colors[np.random.randint(0, len(colors))]),
            thickness=(-1 if with_mask else thickness),
        )
        if with_mask:
            image = cv2.addWeighted(image, 1.0, mask, 0.6, 0)

        if len(bbox) == 5 or len(bbox) == 6:
            cv2.putText(
                image,
                txt,
                (bbox_f[0], bbox_f[1] - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

    return image


def merge_images_with_boundary(images_list, row_col_num=(1, -1)):
    """
    To merge a list of images.

    Additionally, parameter ``row_col_num`` supports user specified merge format.
    Notice, specified format must be greater than or equal to images number.

    Args:
        image_lists: list of np.ndarray.
        row_col_num: merge format. default is (1, -1), image will line up to show.
            when set example=(2, 5), images will display in two rows and five columns.
    """
    if not isinstance(images_list, list):
        images_list = [images_list]

    images_merged = merge_images(images_list, row_col_num)
    return images_merged


def merge_images(images_list, row_col_num):
    """
    Merges all input images as an image with specified merge format.

    Args:
        images_list: images list
        row_col_num: number of rows and columns displayed
    Return:
        image: merges image
    """

    num_images = len(images_list)
    row, col = row_col_num

    assert row > 0 or col > 0, 'row and col cannot be negative at same time!'

    for image in images_list:
        cv2.rectangle(image, (0, 0), (image.shape[1], image.shape[0]), (255, 0, 255))

    if row_col_num[1] < 0 or num_images < row:
        images_merged = np.hstack(images_list)
    elif row_col_num[0] < 0 or num_images < col:
        images_merged = np.vstack(images_list)
    else:
        assert row * col >= num_images, 'Images overboundary, not enough windows to display all images!'

        fill_img_list = [np.zeros(images_list[0].shape, dtype=np.uint8)] * (row * col - num_images)
        images_list.extend(fill_img_list)
        merge_imgs_col = []
        for i in range(row):
            start = col * i
            end = col * (i + 1)
            merge_col = np.hstack(images_list[start: end])
            merge_imgs_col.append(merge_col)

        images_merged = np.vstack(merge_imgs_col)

    return images_merged
