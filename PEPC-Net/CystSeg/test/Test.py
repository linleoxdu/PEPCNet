import argparse
import os
from torch.cuda.amp import autocast
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from CystSeg.models.nnUNet.pepc_net import PEPC_Net
from CystSeg.utils.data_utils import get_loader
from CystSeg.utils.img_utils import dice, resample_3d

parser = argparse.ArgumentParser(description="segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/path_to_dataset", type=str,
                    help="dataset directory")
parser.add_argument("--exp_name", default="cyst_seg_test", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_1.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="/path_to_model_weight",
    type=str,
    help="pretrained model name",
)
parser.add_argument("--feature_size", default=32, type=int, help="feature size")
parser.add_argument("--crop_foreground", action="store_true", help="crop foreground and the inference")
parser.add_argument("--test_time_augmentation", action="store_true", help="flip axis when inference")
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-921.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=2001.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--roi_x", default=160, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=160, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=2, type=int, help="number of workers")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    # setting for cyst segmentation
    stride = [[1, 1, 1],
              [2, 2, 2],
              [2, 2, 2],
              [2, 2, 2],
              [2, 2, 2],
              [2, 2, 1]]
    kwargs = {
        'conv_bias': True,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        "norm_op": nn.InstanceNorm3d,
        'dropout_op': None, 'dropout_op_kwargs': None,
        'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
    }
    model = PEPC_Net(16, args.in_channels, 6, (32, 64, 128, 256, 320, 320), nn.Conv3d, 3, stride,
                     (2, 2, 2, 2, 2, 2),
                     args.out_channels, (2, 2, 2, 2, 2), False, kwargs["norm_op"], kwargs["norm_op_kwargs"],
                     kwargs["dropout_op"], kwargs["dropout_op_kwargs"], kwargs["nonlin"],
                     kwargs["nonlin_kwargs"],
                     deep_supervision=True)

    model_dict = torch.load(pretrained_pth)["state_dict"]
    model.load_state_dict(model_dict)
    model.eval()
    model.to(device)

    with torch.no_grad():
        with autocast(enabled=True):
            case_dices = []
            for i, batch in enumerate(val_loader):
                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
                assert val_labels.shape == val_labels.shape
                original_affine = batch["label_meta_dict"]["affine"][0].numpy()
                _, _, h, w, d = val_labels.shape
                target_shape = (h, w, d)

                if args.crop_foreground:
                    row_shape = batch["image_meta_dict"]["dim"][0][1: 4]
                    foreground_start_coord = batch["foreground_start_coord"][0]
                    foreground_start_padding = foreground_start_coord
                    foreground_end_coord = batch["foreground_end_coord"][0]
                    foreground_end_padding = [(rs - fec).item() for rs, fec in zip(row_shape, foreground_end_coord)]
                    foreground_start_padding = foreground_start_padding.numpy().tolist()
                    padding_coord = [[foreground_start_padding[0], foreground_end_padding[0]],
                                     [foreground_start_padding[1], foreground_end_padding[1]],
                                     [foreground_start_padding[2], foreground_end_padding[2]]]

                img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
                print("Inference on case {}".format(img_name))

                if args.test_time_augmentation:
                    data = [val_inputs,
                            torch.flip(val_inputs, dims=(2,)),
                            torch.flip(val_inputs, dims=(3,)),
                            torch.flip(val_inputs, dims=(4,)),
                            torch.flip(val_inputs, dims=(2, 3)),
                            torch.flip(val_inputs, dims=(2, 4)),
                            torch.flip(val_inputs, dims=(3, 4)),
                            torch.flip(val_inputs, dims=(2, 3, 4))]
                    val_outputs_list = []
                    for i, d in enumerate(data):
                        val_outputs = sliding_window_inference(
                            d, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap,
                            mode="gaussian"
                        )
                        val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                        if i == 0:
                            val_outputs_list.append(val_outputs)
                        elif i == 1 or i == 2 or i == 3:
                            val_outputs_list.append(np.flip(val_outputs, axis=(i + 1,)))
                        elif i == 4:
                            val_outputs_list.append(np.flip(val_outputs, axis=(2, 3)))
                        elif i == 5:
                            val_outputs_list.append(np.flip(val_outputs, axis=(2, 4)))
                        elif i == 6:
                            val_outputs_list.append(np.flip(val_outputs, axis=(3, 4)))
                        else:
                            val_outputs_list.append(np.flip(val_outputs, axis=(2, 3, 4)))

                    val_outputs = np.concatenate(val_outputs_list, axis=0)
                    val_outputs = np.mean(val_outputs, axis=0, keepdims=True)
                    val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                    val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
                else:
                    val_outputs = sliding_window_inference(
                        val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap,
                        mode="gaussian"
                    )
                    val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                    val_outputs = np.argmax(val_outputs, axis=1).astype(np.uint8)[0]
                    val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]

                val_outputs = resample_3d(val_outputs, target_shape)

                case_dice = dice(val_outputs == 1, val_labels == 1)
                case_dices.append(case_dice)
                print("Case Dice: {}".format(case_dice))

                if args.crop_foreground:
                    val_outputs = np.pad(val_outputs, padding_coord)

                nib.save(
                    nib.Nifti1Image(val_outputs.astype(np.uint8), original_affine),
                    os.path.join(output_directory, img_name)
                )

            print("Overall Mean Dice: {}".format(np.mean(case_dices)))


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    main()
