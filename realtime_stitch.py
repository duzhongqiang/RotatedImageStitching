"""
实时全景拼接 — 摄像头水平旋转边拍边拼

操作说明：
  - 程序启动后显示摄像头画面
  - 检测到足够位移时自动采集关键帧并拼接
  - 按 's' 手动强制采集当前帧
  - 按 'q' 退出并保存全景图
  - 按 'r' 重置重新开始

用法：
  python realtime_stitch.py
  python realtime_stitch.py --camera 0 --output result.jpg --focal 800
"""

import argparse
import logging
import time

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class RealtimePanoramaStitcher:
    """实时增量式全景拼接器。

    与离线版的核心区别：
    1. 以第一帧为坐标原点（左锚点），新帧持续向右扩展
    2. 增量拼接：每次只处理新帧与上一关键帧的关系
    3. 关键帧自动检测：位移足够大（有新内容）且重叠足够多（能匹配）才采集

    Parameters
    ----------
    focal_len : float | None
        焦距（像素），None 时用帧宽度的 0.7 倍估算。
    min_shift_ratio : float
        触发关键帧的最小水平位移（相对帧宽的比例），默认 0.08。
    max_shift_ratio : float
        最大允许位移（超过说明转太快、重叠不够），默认 0.6。
    min_matches : int
        匹配点阈值，低于此值认为两帧无法对齐。
    """

    def __init__(
        self,
        focal_len: float | None = None,
        min_shift_ratio: float = 0.08,
        max_shift_ratio: float = 0.60,
        min_matches: int = 10,
        direction: str = "right",
    ):
        if direction not in ("left", "right"):
            raise ValueError("direction 必须是 'left' 或 'right'")
        self.direction = direction
        self.focal_len = focal_len
        self.min_shift_ratio = min_shift_ratio
        self.max_shift_ratio = max_shift_ratio
        self.min_matches = min_matches

        self._sift = cv2.SIFT_create()
        self._flann = cv2.FlannBasedMatcher(
            {"algorithm": 1, "trees": 5}, {"checks": 50}
        )

        self._reset()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def start(self, camera_id: int = 0, output: str = "realtime_panorama.jpg"):
        """启动实时拼接主循环。"""
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {camera_id}")

        logger.info("摄像头已打开，按 's' 手动采帧，'r' 重置，'q' 退出保存")
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Panorama", cv2.WINDOW_NORMAL)

        fps_t = time.time()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("读取摄像头失败")
                break

            # 向左旋转时水平翻转，内部统一按向右处理
            if self.direction == "left":
                frame = cv2.flip(frame, 1)

            frame_count += 1
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                self._reset()
                logger.info("已重置")
            elif key == ord("s"):
                self._add_keyframe(frame, forced=True)

            # 自动关键帧检测（每 3 帧检测一次，降低 CPU 占用）
            if frame_count % 3 == 0:
                self._auto_detect(frame)

            # 显示摄像头画面（叠加状态信息）
            display = frame.copy()
            self._draw_status(display)
            cv2.imshow("Camera", display)

            # 显示当前全景（缩小到 800 宽以便预览）
            if self._panorama is not None:
                preview = self._resize_for_display(self._panorama, max_w=1200)
                cv2.imshow("Panorama", preview)

            # FPS 计算
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_t)
                fps_t = time.time()
                logger.debug("FPS: %.1f  关键帧数: %d", fps, len(self._keyframe_list))

        cap.release()
        cv2.destroyAllWindows()

        if self._panorama is not None:
            # 裁剪黑边后保存
            result = self._crop_black_border(self._panorama)
            # 左旋时翻转还原，使全景图与实际方向一致
            if self.direction == "left":
                result = cv2.flip(result, 1)
            cv2.imwrite(output, result)
            h, w = result.shape[:2]
            logger.info("全景图已保存：%s  (%dx%d)  共 %d 帧",
                        output, w, h, len(self._keyframe_list))
        else:
            logger.warning("未采集到足够帧，未保存")

    # ------------------------------------------------------------------
    # 内部状态管理
    # ------------------------------------------------------------------

    def _reset(self):
        self._panorama: np.ndarray | None = None   # 当前全景图（BGR float32）
        self._weight_map: np.ndarray | None = None  # 对应的权重图
        self._last_kp = None       # 上一关键帧的 SIFT 关键点
        self._last_desc = None     # 上一关键帧的描述符
        self._last_frame = None    # 上一关键帧原图
        self._H_cumulative = None  # 累积单应矩阵（最新帧 → 全景坐标系）
        self._focal: float | None = None
        self._canvas_offset = np.array([0, 0], dtype=np.float64)  # 画布原点偏移
        self._keyframe_list: list = []

    # ------------------------------------------------------------------
    # 关键帧检测与采集
    # ------------------------------------------------------------------

    def _auto_detect(self, frame: np.ndarray):
        """检测当前帧是否应作为新关键帧加入全景。"""
        if self._last_frame is None:
            # 第一帧，直接初始化
            self._add_keyframe(frame, forced=True)
            return

        h, w = frame.shape[:2]
        kp, desc = self._detect(frame)
        matches = self._match(self._last_desc, desc)

        if len(matches) < self.min_matches:
            return  # 匹配不上，等待

        # 估计两帧间的水平位移
        H = self._homography(self._last_kp, kp, matches)
        if H is None:
            return

        # 取单应矩阵对图像中心的位移作为判断依据
        cx, cy = w / 2.0, h / 2.0
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        pt_t = cv2.perspectiveTransform(pt, H)[0][0]
        # 正值表示摄像头向右转，负值向左转；左旋帧已被翻转，此处始终期望正值
        signed_dx = pt_t[0] - cx
        if signed_dx <= 0:
            return  # 摄像头向反方向移动，丢弃该帧
        dx = signed_dx

        min_px = w * self.min_shift_ratio
        max_px = w * self.max_shift_ratio

        if dx < min_px:
            return  # 位移太小，还没转到位
        if dx > max_px:
            logger.warning("位移过大 (%.0fpx)，可能转太快导致重叠不足", dx)
            return

        self._add_keyframe(frame, forced=False, H_to_prev=H, kp=kp, desc=desc)

    def _add_keyframe(
        self,
        frame: np.ndarray,
        forced: bool = False,
        H_to_prev: np.ndarray | None = None,
        kp=None,
        desc=None,
    ):
        """将当前帧加入全景。"""
        h, w = frame.shape[:2]

        # 初始化焦距
        if self._focal is None:
            self._focal = self.focal_len if self.focal_len is not None else w * 0.7
            logger.info("焦距 f=%.1f 像素", self._focal)

        # 柱面投影
        warped_frame = self._cylindrical_warp(frame, self._focal)

        # 提取特征（如果还没提取）
        if kp is None or desc is None:
            kp, desc = self._detect(warped_frame)
        else:
            # 重新在柱面图上提取（auto_detect 里用的是原图特征，这里换成柱面图）
            kp, desc = self._detect(warped_frame)

        # ---- 第一帧：直接作为初始全景 ----
        if self._panorama is None:
            self._panorama = warped_frame.astype(np.float32)
            self._weight_map = self._make_weight(warped_frame)
            self._H_cumulative = np.eye(3, dtype=np.float64)
            self._last_kp, self._last_desc, self._last_frame = kp, desc, warped_frame
            self._keyframe_list.append(warped_frame)
            logger.info("初始帧已设置，全景尺寸: %dx%d", w, h)
            return

        # ---- 后续帧：计算与上一关键帧的单应性 ----
        if H_to_prev is None or forced:
            matches = self._match(self._last_desc, desc)
            if len(matches) < self.min_matches:
                logger.warning("强制采帧：匹配点不足 (%d)，跳过", len(matches))
                return
            H_to_prev = self._homography(self._last_kp, kp, matches)
            if H_to_prev is None:
                logger.warning("单应性计算失败，跳过")
                return

        # 累积单应矩阵：新帧 → 第一帧坐标系
        # H_to_prev 将新帧坐标映射到上一帧，H_cumulative 将上一帧映射到全景
        H_new_abs = self._H_cumulative @ H_to_prev

        # 计算新帧四角在全景坐标系中的位置，动态扩展画布
        ch, cw = self._panorama.shape[:2]
        wh, ww = warped_frame.shape[:2]

        corners = np.float32([[0, 0], [ww, 0], [ww, wh], [0, wh]]).reshape(-1, 1, 2)
        T_offset = np.array([[1, 0, self._canvas_offset[0]],
                              [0, 1, self._canvas_offset[1]],
                              [0, 0, 1]], dtype=np.float64)
        H_with_offset = T_offset @ H_new_abs
        corners_t = cv2.perspectiveTransform(corners, H_with_offset)

        all_x = np.append(corners_t[:, 0, 0], [0, cw])
        all_y = np.append(corners_t[:, 0, 1], [0, ch])
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        # 需要向左/上扩展时调整偏移
        pad_left = max(0, int(np.ceil(-x_min)))
        pad_top  = max(0, int(np.ceil(-y_min)))
        new_w = int(np.ceil(x_max)) + pad_left + 1
        new_h = int(np.ceil(y_max)) + pad_top  + 1

        if pad_left > 0 or pad_top > 0:
            # 平移现有画布
            T_shift = np.array([[1, 0, pad_left],
                                 [0, 1, pad_top],
                                 [0, 0, 1]], dtype=np.float64)
            self._panorama = cv2.warpPerspective(
                self._panorama.astype(np.uint8), T_shift, (new_w, new_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            ).astype(np.float32)
            self._weight_map = cv2.warpPerspective(
                self._weight_map, T_shift, (new_w, new_h),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
            )
            self._canvas_offset += np.array([pad_left, pad_top], dtype=np.float64)
        else:
            # 只扩展宽高
            if new_w > cw or new_h > ch:
                pad_canvas = np.zeros((new_h, new_w, 3), dtype=np.float32)
                pad_canvas[:ch, :cw] = self._panorama
                self._panorama = pad_canvas
                pad_weight = np.zeros((new_h, new_w), dtype=np.float32)
                pad_weight[:ch, :cw] = self._weight_map
                self._weight_map = pad_weight

        # 将新帧变换到全景坐标系并融合
        T_final = np.array([[1, 0, self._canvas_offset[0]],
                             [0, 1, self._canvas_offset[1]],
                             [0, 0, 1]], dtype=np.float64)
        H_final = T_final @ H_new_abs
        ph, pw = self._panorama.shape[:2]

        warped = cv2.warpPerspective(
            warped_frame, H_final, (pw, ph),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        new_weight = cv2.warpPerspective(
            self._make_weight(warped_frame), H_final, (pw, ph),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )

        # 羽化融合
        mask = new_weight > 0
        self._panorama[mask] = (
            self._panorama[mask] * self._weight_map[mask, np.newaxis] +
            warped.astype(np.float32)[mask] * new_weight[mask, np.newaxis]
        ) / (self._weight_map[mask] + new_weight[mask] + 1e-6)[:, np.newaxis]
        self._weight_map = np.maximum(self._weight_map, new_weight)

        # 更新状态
        self._H_cumulative = H_new_abs
        self._last_kp, self._last_desc, self._last_frame = kp, desc, warped_frame
        self._keyframe_list.append(warped_frame)
        logger.info("关键帧 #%d 已拼接，当前全景: %dx%d",
                    len(self._keyframe_list), pw, ph)

    # ------------------------------------------------------------------
    # 特征检测 / 匹配 / 单应性（与离线版相同）
    # ------------------------------------------------------------------

    def _detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self._sift.detectAndCompute(gray, None)

    def _match(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        matches = self._flann.knnMatch(desc1, desc2, k=2)
        return [m for m, n in matches if m.distance < 0.75 * n.distance]

    def _homography(self, kp1, kp2, matches):
        if len(matches) < self.min_matches:
            return None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
        return H

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def _cylindrical_warp(self, img, f):
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        xd, yd = np.meshgrid(np.arange(w, dtype=np.float32),
                              np.arange(h, dtype=np.float32))
        xs = f * np.tan((xd - cx) / f) + cx
        ys = (yd - cy) * np.sqrt(((xd - cx) / f) ** 2 + 1) + cy
        return cv2.remap(img, xs, ys, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def _make_weight(self, img):
        """用距离变换生成权重图。"""
        mask = (img.sum(axis=2) > 0).astype(np.uint8)
        return cv2.distanceTransform(mask, cv2.DIST_L2, 5).astype(np.float32)

    def _crop_black_border(self, img):
        gray = cv2.cvtColor(img.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img.clip(0, 255).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))
        return img[y:y+h, x:x+w].clip(0, 255).astype(np.uint8)

    @staticmethod
    def _resize_for_display(img, max_w=1200):
        h, w = img.shape[:2]
        if w <= max_w:
            return img.clip(0, 255).astype(np.uint8)
        scale = max_w / w
        return cv2.resize(img.clip(0, 255).astype(np.uint8),
                          (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    def _draw_status(self, frame):
        n = len(self._keyframe_list)
        status = f"Keyframes: {n} | 's':capture 'r':reset 'q':quit"
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="实时全景拼接")
    p.add_argument("--camera", "-c", type=int, default=0, help="摄像头 ID（默认 0）")
    p.add_argument("--output", "-o", default="realtime_panorama.jpg", help="输出路径")
    p.add_argument("--focal", "-f", type=float, default=None, help="焦距（像素）")
    p.add_argument("--min-shift", type=float, default=0.08,
                   help="触发关键帧的最小位移比例（默认 0.08）")
    p.add_argument("--direction", "-d", choices=["left", "right"], default="right",
                   help="旋转方向：right（向右，默认）或 left（向左）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stitcher = RealtimePanoramaStitcher(
        focal_len=args.focal,
        min_shift_ratio=args.min_shift,
        direction=args.direction,
    )
    stitcher.start(camera_id=args.camera, output=args.output)
