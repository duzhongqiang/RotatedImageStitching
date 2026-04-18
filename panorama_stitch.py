"""
旋转图像全景拼接 — 核心实现
算法流程：加载图像 → 柱面投影 → SIFT特征检测 → FLANN匹配 → RANSAC单应性 → 链式变换 → 羽化融合
"""

import os
import math
import logging
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


class PanoramaStitcher:
    """水平旋转拍摄的序列图像全景拼接器。

    Parameters
    ----------
    focal_len : float or None
        相机焦距（像素）。None 时自动用图像宽度的 0.7 倍估算。
    use_cylindrical : bool
        是否在匹配前先做柱面投影（旋转幅度大于 ~45° 时推荐开启）。
    lowe_ratio : float
        Lowe's ratio test 阈值，越小匹配越严格，默认 0.75。
    min_match_count : int
        计算单应性所需的最少匹配点数，默认 10。
    """

    def __init__(
        self,
        focal_len: float | None = None,
        use_cylindrical: bool = True,
        lowe_ratio: float = 0.75,
        min_match_count: int = 10,
    ):
        self.focal_len = focal_len
        self.use_cylindrical = use_cylindrical
        self.lowe_ratio = lowe_ratio
        self.min_match_count = min_match_count

        # SIFT 检测器
        self._sift = cv2.SIFT_create()

        # FLANN 匹配器（针对 SIFT float 描述符）
        index_params = {"algorithm": 1, "trees": 5}   # FLANN_INDEX_KDTREE = 1
        search_params = {"checks": 50}
        self._flann = cv2.FlannBasedMatcher(index_params, search_params)

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def run(self, input_folder: str, output_path: str = "panorama.jpg") -> np.ndarray:
        """完整拼接流程。

        Parameters
        ----------
        input_folder : str
            存放序列图像的文件夹路径（按文件名自然排序）。
        output_path : str
            输出全景图保存路径。

        Returns
        -------
        np.ndarray
            拼接好的 BGR 全景图。
        """
        images = self.load_images(input_folder)
        if len(images) < 2:
            raise ValueError(f"至少需要 2 张图像，当前只找到 {len(images)} 张。")

        logger.info("共加载 %d 张图像", len(images))

        # 如果未指定焦距，用第一张图宽度的 0.7 倍估算
        h0, w0 = images[0].shape[:2]
        focal = self.focal_len if self.focal_len is not None else w0 * 0.7
        logger.info("使用焦距 f=%.1f 像素", focal)

        # 可选：柱面投影预处理
        if self.use_cylindrical:
            logger.info("执行柱面投影...")
            images = [self.cylindrical_warp(img, focal) for img in images]

        # 拼接所有图像
        panorama = self.warp_and_blend(images, focal)

        # 裁剪多余黑边
        panorama = self._crop_black_border(panorama)

        # 保存
        cv2.imwrite(output_path, panorama)
        logger.info("全景图已保存至 %s  (尺寸: %dx%d)", output_path, panorama.shape[1], panorama.shape[0])
        return panorama

    # ------------------------------------------------------------------
    # 图像加载
    # ------------------------------------------------------------------

    def load_images(self, folder: str) -> list[np.ndarray]:
        """从文件夹中按自然顺序加载所有图像。"""
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

    # ------------------------------------------------------------------
    # 柱面投影
    # ------------------------------------------------------------------

    def cylindrical_warp(self, img: np.ndarray, f: float) -> np.ndarray:
        """将图像映射到柱面坐标，消除大角度旋转的直线弯曲失真。

        Parameters
        ----------
        img : np.ndarray
            输入 BGR 图像。
        f : float
            焦距（像素）。

        Returns
        -------
        np.ndarray
            柱面投影后的图像（背景为黑色）。
        """
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # 建立目标坐标网格
        xd, yd = np.meshgrid(np.arange(w, dtype=np.float32),
                              np.arange(h, dtype=np.float32))

        # 柱面坐标 → 原始像素坐标（逆映射）
        xs = f * np.tan((xd - cx) / f) + cx
        ys = (yd - cy) * np.sqrt(((xd - cx) / f) ** 2 + 1) + cy

        warped = cv2.remap(
            img, xs, ys,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return warped

    # ------------------------------------------------------------------
    # 特征检测与匹配
    # ------------------------------------------------------------------

    def detect_features(self, img: np.ndarray):
        """SIFT 关键点 + 描述符提取。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = self._sift.detectAndCompute(gray, None)
        return kp, desc

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        """FLANN kNN(k=2) 匹配 + Lowe's ratio test。"""
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        matches = self._flann.knnMatch(desc1, desc2, k=2)
        good = [m for m, n in matches if m.distance < self.lowe_ratio * n.distance]
        return good

    # ------------------------------------------------------------------
    # 单应性估计
    # ------------------------------------------------------------------

    def compute_homography(self, kp1, kp2, matches) -> np.ndarray | None:
        """用 RANSAC 从匹配点对估计单应矩阵 H（将 img2 映射到 img1 坐标系）。"""
        if len(matches) < self.min_match_count:
            logger.warning("匹配点 %d 少于阈值 %d，跳过该图像对", len(matches), self.min_match_count)
            return None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransacReprojThreshold=4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        logger.info("  匹配点: %d  内点: %d  单应性: %s",
                    len(matches), inliers, "成功" if H is not None else "失败")
        return H

    # ------------------------------------------------------------------
    # 链式变换 + 融合
    # ------------------------------------------------------------------

    def warp_and_blend(self, images: list[np.ndarray], focal: float) -> np.ndarray:
        """将所有图像变换到以中间帧为参考系的统一画布上并融合。

        以中间图像为参考系，向两侧分别累乘单应矩阵，误差分布更均匀。
        """
        n = len(images)
        ref_idx = n // 2  # 参考帧索引

        # 1. 计算所有相邻帧的单应矩阵 H[i] = img[i+1] → img[i]
        H_pairs: list[np.ndarray | None] = []
        for i in range(n - 1):
            logger.info("计算帧 %d ↔ %d 的单应矩阵...", i, i + 1)
            kp1, d1 = self.detect_features(images[i])
            kp2, d2 = self.detect_features(images[i + 1])
            matches = self.match_features(d1, d2)
            H = self.compute_homography(kp1, kp2, matches)
            H_pairs.append(H)

        # 2. 以 ref_idx 为基准，链式累乘得到每帧到参考帧的变换 H_abs[i]
        H_abs: list[np.ndarray | None] = [None] * n
        H_abs[ref_idx] = np.eye(3, dtype=np.float64)

        # 向右扩展（ref_idx → n-1）
        for i in range(ref_idx, n - 1):
            if H_pairs[i] is None:
                logger.warning("帧 %d↔%d 单应性缺失，使用前一帧变换", i, i + 1)
                H_abs[i + 1] = H_abs[i]
            else:
                # H_pairs[i]: img[i+1]→img[i], 再乘 H_abs[i]: img[i]→ref
                H_abs[i + 1] = H_abs[i] @ H_pairs[i]

        # 向左扩展（ref_idx → 0）
        for i in range(ref_idx, 0, -1):
            if H_pairs[i - 1] is None:
                logger.warning("帧 %d↔%d 单应性缺失，使用后一帧变换", i - 1, i)
                H_abs[i - 1] = H_abs[i]
            else:
                # inv(H_pairs[i-1]): img[i-1]→img[i], 再乘 H_abs[i]: img[i]→ref
                H_abs[i - 1] = H_abs[i] @ np.linalg.inv(H_pairs[i - 1])

        # 3. 计算画布尺寸
        canvas_size, offset = self._compute_canvas(images, H_abs)
        cw, ch = canvas_size  # (width, height)

        # 4. 将 offset 平移合并到各单应矩阵
        T = np.array([[1, 0, offset[0]],
                      [0, 1, offset[1]],
                      [0, 0, 1]], dtype=np.float64)
        H_final = [T @ H if H is not None else T for H in H_abs]

        # 5. 逐帧变换并羽化融合
        canvas = np.zeros((ch, cw, 3), dtype=np.float32)
        weight_sum = np.zeros((ch, cw), dtype=np.float32)

        for i, (img, H) in enumerate(zip(images, H_final)):
            logger.info("融合第 %d/%d 帧...", i + 1, n)
            warped = cv2.warpPerspective(img, H, (cw, ch),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=0)
            # 有效像素 mask
            mask = (warped.sum(axis=2) > 0).astype(np.float32)
            # 距离变换作为权重（离边缘越近权重越低，内部更高）
            weight = cv2.distanceTransform(
                mask.astype(np.uint8), cv2.DIST_L2, 5
            )
            weight = weight.astype(np.float32)

            canvas += warped.astype(np.float32) * weight[:, :, np.newaxis]
            weight_sum += weight

        # 避免除零
        weight_sum = np.maximum(weight_sum, 1e-6)
        panorama = (canvas / weight_sum[:, :, np.newaxis]).clip(0, 255).astype(np.uint8)

        return panorama

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _compute_canvas(
        self, images: list[np.ndarray], H_abs: list[np.ndarray | None]
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        """根据各图四角变换结果计算画布尺寸和偏移量。"""
        all_corners = []
        for img, H in zip(images, H_abs):
            if H is None:
                continue
            h, w = img.shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(corners, H)
            all_corners.append(transformed)

        all_corners = np.concatenate(all_corners, axis=0)
        x_min, y_min = all_corners[:, 0, 0].min(), all_corners[:, 0, 1].min()
        x_max, y_max = all_corners[:, 0, 0].max(), all_corners[:, 0, 1].max()

        offset_x = int(math.ceil(-x_min)) if x_min < 0 else 0
        offset_y = int(math.ceil(-y_min)) if y_min < 0 else 0
        canvas_w = int(math.ceil(x_max)) + offset_x + 1
        canvas_h = int(math.ceil(y_max)) + offset_y + 1

        logger.info("画布尺寸: %dx%d  偏移: (%d, %d)", canvas_w, canvas_h, offset_x, offset_y)
        return (canvas_w, canvas_h), (offset_x, offset_y)

    def _crop_black_border(self, img: np.ndarray) -> np.ndarray:
        """裁剪图像四周的黑边。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img
        # 取最大连通域的包围矩形
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return img[y: y + h, x: x + w]

    @staticmethod
    def _natural_sort_key(s: str):
        """自然排序键，使 img2.jpg 排在 img10.jpg 前面。"""
        import re
        parts = re.split(r"(\d+)", s)
        return [int(p) if p.isdigit() else p.lower() for p in parts]
