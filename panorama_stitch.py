"""
旋转图像全景拼接 — 核心实现
使用 OpenCV Stitcher：球面/平面投影 + Bundle Adjustment + 多频段融合
"""

import re
import math
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_STITCH_ERRORS = {
    cv2.Stitcher_ERR_NEED_MORE_IMGS: "图像数量不足，请增加输入图像",
    cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "单应矩阵估计失败，图像重叠区域可能不足",
    cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "相机参数 Bundle Adjustment 失败",
}


class PanoramaStitcher:
    """旋转拍摄序列图像全景拼接器。

    Parameters
    ----------
    mode : str
        拼接模式：
        - "panorama"：相机原地旋转（偏航/俯仰/滚转均支持），使用球面投影，适合大角度旋转。
        - "scans"：相机平移扫描，使用仿射变换，适合文档/平面场景。
    keyframe_shift : float
        关键帧筛选阈值（像素）：相邻帧相位相关位移超过该值才纳入拼接，0=不筛选。
    """

    def __init__(self, mode: str = "panorama", keyframe_shift: float = 0.0):
        self.mode = mode
        self.keyframe_shift = keyframe_shift

    def run(self, input_folder: str, output_path: str = "panorama.jpg") -> np.ndarray:
        images = self.load_images(input_folder)
        if len(images) < 2:
            raise ValueError(f"至少需要 2 张图像，当前只找到 {len(images)} 张。")

        logger.info("共加载 %d 张图像", len(images))

        if self.keyframe_shift > 0:
            images = self._filter_keyframes(images)
            logger.info("关键帧筛选后: %d 张 (阈值 %.0fpx)", len(images), self.keyframe_shift)

        stitcher_mode = cv2.Stitcher_PANORAMA if self.mode == "panorama" else cv2.Stitcher_SCANS
        stitcher = cv2.Stitcher_create(stitcher_mode)

        logger.info("开始拼接（模式: %s）...", self.mode)
        status, result = stitcher.stitch(images)

        if status != cv2.Stitcher_OK:
            msg = _STITCH_ERRORS.get(status, f"未知错误 (code={status})")
            raise RuntimeError(msg)

        cv2.imwrite(output_path, result)
        logger.info("全景图已保存至 %s  (尺寸: %dx%d)", output_path, result.shape[1], result.shape[0])
        return result

    def load_images(self, folder: str) -> list:
        folder_path = Path(folder)
        if not folder_path.is_dir():
            raise FileNotFoundError(f"目录不存在: {folder}")

        files = sorted(
            [p for p in folder_path.iterdir() if p.suffix.lower() in SUPPORTED_EXT],
            key=lambda p: self._natural_sort_key(p.name),
        )
        if not files:
            raise FileNotFoundError(f"目录 {folder} 中未找到支持的图像文件。")

        images = []
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                logger.warning("无法读取: %s，已跳过", f.name)
                continue
            images.append(img)
            logger.info("  已加载: %s  (%dx%d)", f.name, img.shape[1], img.shape[0])

        return images

    def _filter_keyframes(self, images: list) -> list:
        """相位相关法预过滤冗余帧，保留位移超过阈值的关键帧。"""
        selected = [images[0]]
        last = images[0]
        for img in images[1:]:
            g1 = cv2.cvtColor(last, cv2.COLOR_BGR2GRAY).astype(np.float32)
            g2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            (dx, dy), _ = cv2.phaseCorrelate(g1, g2)
            if math.hypot(dx, dy) >= self.keyframe_shift:
                selected.append(img)
                last = img
        return selected

    @staticmethod
    def _natural_sort_key(s: str):
        parts = re.split(r"(\d+)", s)
        return [int(p) if p.isdigit() else p.lower() for p in parts]
