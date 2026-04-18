"""
命令行入口：旋转图像全景拼接

用法示例：
    python run_stitch.py --input ./images --output panorama.jpg
    python run_stitch.py --input ./images --output panorama.jpg --focal 800
    python run_stitch.py --input ./images --output panorama.jpg --no-cylindrical
"""

import argparse
import sys
import time

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description="水平旋转摄像头序列图像全景拼接",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基本用法（自动估算焦距）
  python run_stitch.py --input images/ --output result.jpg

  # 指定相机焦距（像素，提高精度）
  python run_stitch.py --input images/ --output result.jpg --focal 800

  # 关闭柱面投影（适合旋转角度较小的场景）
  python run_stitch.py --input images/ --output result.jpg --no-cylindrical

  # 调整匹配严格程度（值越小越严格，0.6~0.85 之间）
  python run_stitch.py --input images/ --output result.jpg --ratio 0.7
""",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="输入图像文件夹路径（按文件名顺序排列）",
    )
    parser.add_argument(
        "--output", "-o",
        default="panorama.jpg",
        help="输出全景图路径（默认：panorama.jpg）",
    )
    parser.add_argument(
        "--focal", "-f",
        type=float,
        default=None,
        help="相机焦距（像素）。不指定时自动估算为图像宽度的 0.7 倍",
    )
    parser.add_argument(
        "--no-cylindrical",
        action="store_true",
        help="关闭柱面投影预处理（仅适合小角度旋转场景）",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.75,
        help="Lowe's ratio test 阈值（默认：0.75，范围 0.5~0.9）",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=10,
        help="计算单应性所需最少匹配点数（默认：10）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 延迟导入，方便在缺少依赖时给出清晰提示
    try:
        from panorama_stitch import PanoramaStitcher
    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        print("请先安装依赖：pip install -r requirements.txt")
        sys.exit(1)

    stitcher = PanoramaStitcher(
        focal_len=args.focal,
        use_cylindrical=not args.no_cylindrical,
        lowe_ratio=args.ratio,
        min_match_count=args.min_matches,
    )

    print(f"输入目录 : {args.input}")
    print(f"输出路径 : {args.output}")
    print(f"柱面投影 : {'关闭' if args.no_cylindrical else '开启'}")
    print(f"焦距     : {args.focal if args.focal else '自动估算'}")
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
