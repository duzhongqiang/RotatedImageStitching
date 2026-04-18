"""
命令行入口：旋转图像全景拼接

用法示例：
    python run_stitch.py --input ./images --output panorama.jpg
    python run_stitch.py --input ./images --output panorama.jpg --mode scans
    python run_stitch.py --input ./images --output panorama.jpg --keyframe-shift 50
"""

import argparse
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="旋转摄像头序列图像全景拼接（OpenCV Stitcher）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 旋转拍摄（偏航/俯仰/滚转均支持，推荐）
  python run_stitch.py --input images/ --output result.jpg

  # 平移扫描场景（文档、平面）
  python run_stitch.py --input images/ --output result.jpg --mode scans

  # 预过滤冗余帧（帧数多时加速）
  python run_stitch.py --input images/ --output result.jpg --keyframe-shift 50
""",
    )
    parser.add_argument("--input", "-i", required=True, help="输入图像文件夹路径")
    parser.add_argument("--output", "-o", default="panorama.jpg", help="输出全景图路径（默认：panorama.jpg）")
    parser.add_argument(
        "--mode",
        choices=["panorama", "scans"],
        default="panorama",
        help="拼接模式：panorama=旋转拍摄（默认），scans=平移扫描",
    )
    parser.add_argument(
        "--keyframe-shift",
        type=float,
        default=0.0,
        help="关键帧筛选阈值（像素），相邻帧位移超过此值才纳入拼接，0=不筛选（默认）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from panorama_stitch import PanoramaStitcher
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        print("请先安装依赖：pip install -r requirements.txt")
        sys.exit(1)

    stitcher = PanoramaStitcher(
        mode=args.mode,
        keyframe_shift=args.keyframe_shift,
    )

    print(f"输入目录 : {args.input}")
    print(f"输出路径 : {args.output}")
    print(f"拼接模式 : {args.mode}")
    print(f"关键帧阈值: {args.keyframe_shift if args.keyframe_shift > 0 else '不筛选'}")
    print("-" * 40)

    t0 = time.time()
    try:
        result = stitcher.run(args.input, args.output)
    except FileNotFoundError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[错误] {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[错误] 拼接失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[错误] 拼接失败: {e}")
        raise

    elapsed = time.time() - t0
    h, w = result.shape[:2]
    print("-" * 40)
    print(f"拼接完成！耗时: {elapsed:.1f}s  全景图尺寸: {w}x{h}")
    print(f"结果已保存至: {args.output}")


if __name__ == "__main__":
    main()
