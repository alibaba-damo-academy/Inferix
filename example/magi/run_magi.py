import argparse
import sys

from inferix.pipeline.magi.pipeline import MagiPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run MagiPipeline with different modes.")
    parser.add_argument('--config_file', type=str, help='Path to the configuration file.')
    parser.add_argument(
        '--mode', type=str, choices=['t2v', 'i2v', 'v2v'], required=True, help='Mode to run: t2v, i2v, or v2v.'
    )
    parser.add_argument('--prompt', type=str, required=True, help='Prompt for the pipeline.')
    parser.add_argument('--image_path', type=str, help='Path to the image file (for i2v mode).')
    parser.add_argument('--prefix_video_path', type=str, help='Path to the prefix video file (for v2v mode).')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output video.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    pipeline = MagiPipeline(args.config_file)

    if args.mode == 't2v':
        pipeline.run_text_to_video(prompt=args.prompt, output_path=args.output_path)
    elif args.mode == 'i2v':
        if not args.image_path:
            print("Error: --image_path is required for i2v mode.")
            sys.exit(1)
        pipeline.run_image_to_video(prompt=args.prompt, image_path=args.image_path, output_path=args.output_path)
    elif args.mode == 'v2v':
        if not args.prefix_video_path:
            print("Error: --prefix_video_path is required for v2v mode.")
            sys.exit(1)
        pipeline.run_video_to_video(prompt=args.prompt, prefix_video_path=args.prefix_video_path, output_path=args.output_path)


if __name__ == "__main__":
    main()
