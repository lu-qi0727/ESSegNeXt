# # Copyright (c) OpenMMLab. All rights reserved.




from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os

def main():
    parser = ArgumentParser()
    parser.add_argument('img', nargs='?', default='/hy-tmp/mmsegmentation-main/data/building_datasets/img_dir/train/11_9_31.tif', help='Image file')
    parser.add_argument('config', nargs='?', default='/hy-tmp/mmsegmentation-main/configs/twins/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py', help='Config file')
    parser.add_argument('checkpoint', nargs='?', default='/hy-tmp/mmsegmentation-main/work_dirs/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512/iter_1800.pth', help='Checkpoint file')
    parser.add_argument('--out-dir', nargs='?', default='/hy-tmp/mmsegmentation-main/work_dirs/result.png', help='Directory to save the visualized results')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--with-labels', action='store_true', default=False, help='Whether to display the class labels.')
    parser.add_argument('--title', default='result', help='The image identifier.')
    args = parser.parse_args()

    # Build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # Test a single image
    result = inference_model(model, args.img)
    if result is None:
        raise ValueError("Inference failed. The result is None.")

    # Ensure the output directory exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Construct the output file path
    out_file = os.path.join(args.out_dir, 'result.png')

    # Show the results
    show_result_pyplot(
        model,
        args.img,
        result,
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False,
        out_file=out_file,
        save_dir=args.out_dir  # Add save_dir argument
    )


if __name__ == '__main__':
    main()


# from argparse import ArgumentParser
#
# from mmengine.model import revert_sync_batchnorm
#
# from mmseg.apis import inference_model, init_model, show_result_pyplot
#
#
# def main():
#     parser = ArgumentParser()
#     parser.add_argument('img',nargs='?',default='/hy-tmp/mmsegmentation-main/data/building_datasets/img_dir/train/11_11_21.tif', help='Image file')
#     parser.add_argument('config',nargs='?',default='/hy-tmp/mmsegmentation-main/configs/twins/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512.py', help='Config file')
#     parser.add_argument('checkpoint',nargs='?',default='/hy-tmp/mmsegmentation-main/work_dirs/twins_pcpvt-s_fpn_fpnhead_8xb4-80k_ade20k-512x512/iter_1800.pth', help='Checkpoint file')
#     parser.add_argument('--out-file', nargs='?',default=None, help='Path to output file')
#     parser.add_argument(
#         '--device', default='cuda:0', help='Device used for inference')
#     parser.add_argument(
#         '--opacity',
#         type=float,
#         default=0.5,
#         help='Opacity of painted segmentation map. In (0, 1] range.')
#     parser.add_argument(
#         '--with-labels',
#         action='store_true',
#         default=False,
#         help='Whether to display the class labels.')
#     parser.add_argument(
#         '--title', default='result', help='The image identifier.')
#     args = parser.parse_args()
#
#     # build the model from a config file and a checkpoint file
#     model = init_model(args.config, args.checkpoint, device=args.device)
#     if args.device == 'cpu':
#         model = revert_sync_batchnorm(model)
#     # test a single image
#     result = inference_model(model, args.img)
#     # show the results
#     show_result_pyplot(
#         model,
#         args.img,
#         result,
#         title=args.title,
#         opacity=args.opacity,
#         with_labels=args.with_labels,
#         draw_gt=False,
#         show=False if args.out_file is not None else True,
#         out_file=args.out_file)
#
#
# if __name__ == '__main__':
#     main()
