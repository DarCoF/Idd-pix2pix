import argparse

# Training settings
parser = argparse.ArgumentParser(description='pix2pix-pytorch-implementation')
# DTT In the original code, dataset is required. We don't need it for the Inria Aerial Image Labelling Dataset
parser.add_argument('--dataset', required=False, help='facades')
parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--direction', type=str, default='a2b', help='a2b or b2a')
parser.add_argument('--input_nc', type=int, default=3, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
# Training epochs are defined by range(opt.epoch_count, opt.niter + opt.niter_decay + 1)
# So, originally, the training script epochs from 1 to 201, which takes too long at the beginning
# niter and niter_decay are changed to shorten the amount of time during development
parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')
parser.add_argument('--niter', type=int, default=10, help='# of iter at starting learning rate') # 
parser.add_argument('--niter_decay', type=int, default=1, help='# of iter to linearly decay learning rate to zero') # 100
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam') # 
parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau|cosine')
parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective') # 

# Size of the side of the image. Only square images are accepted
parser.add_argument('--input_image_size', type=int, default=286, help='input image side size')
# Size of the side of the image to be fed to the model. It will generate a squared image
parser.add_argument('--model_side_size', type=int, default=256, help='image fed to the model side size')

parser.add_argument('--source_path', type=str, default='/Datasets/Aerialdataset/train', help='source path from current directory')
parser.add_argument('--dest_train', type=str, default='train', help='destination path from current directory')
parser.add_argument('--dest_valid', type=str, default='test', help='destination path from current directory')
parser.add_argument('--side_size', type=int, default=286, help='size of every side (square output)')
parser.add_argument('--crop_size', type=int, default=1000, help='size of every image crop side')
parser.add_argument('--gt_dir', type=str, default='/gt', help='ground truth image masks folder specific path')
parser.add_argument('--images_dir', type=str, default='/images', help='images folder specific path')
parser.add_argument('--log_dir', type=str, default='log', help='tensorboard log files specific path')
parser.add_argument('--tensor_ext', type=str, default='.npy', help='tensor data extension')
parser.add_argument('--split_pattern', type=str, default='[0-6].tif', help='filtering pattern for train/validation split')

# Activate or deactivate the use of Tensorboard
parser.add_argument('--tb_active', type=bool, default=True, help='should tensorboard be used') # Deactivate for deep trainings
# Which original image should be stored in Tensorboard.
# Inria satellite images are 5000x5000 and consume much CPU and memory, so only
# one image is saved to avoid using too many resources
parser.add_argument('--tb_image', type=str, default='vienna1.npy', help='image to store in tensorboard')
# Number of images saved to tensorboard. Only tb_image will be saved, so the progress
# of generated images can be seen throw epochs. 5 images in 100 epochs means one
# tb_image will be saved every 20 epochs.
parser.add_argument('--tb_number_img', type=int, default=5, help='number of images saved to tensorboard')
# Level of debug (cell output)
parser.add_argument('--debug', type=int, default=0, help='level of debug from 0 (no debug) to 2 (verbose)')
# Number of iteration messages per epoch. They have the form
# ===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}
parser.add_argument('--iter_messages', type=int, default=4, help='number of output messages per epoch')
# Number of epochs to save a checkpoint
parser.add_argument('--checkpoint_epochs', type=int, default=50, help='number of epochs to save a checkpoint')
# Stop training after checkpoint is saved. Useful in long trainings
parser.add_argument('--stop_after_checkpoint', type=int, default=1, help='stop training after a checkpoint has been saved')
# Determine if networks and optimizers should be loaded from a previous training
parser.add_argument('--load_pretrained', type=int, default=0, help='load pretrained networks and optimizers')

# As stated in https://stackoverflow.com/questions/48796169/how-to-fix-ipykernel-launcher-py-error-unrecognized-arguments-in-jupyter
# at least an empty list must be passed to simulate a script execution with no parameters.
# If no parameter is provided, parse_args tries to read _sys.argv[1:], which is not defined
# in a colab execution

# training_args = ['--cuda',
#                  '--epoch_count=1',
#                  '--niter=100',
#                  '--niter_decay=50',
#                  '--lr=0.002',
#                  '--lamb=10',
#                  '--direction=a2b',
#                  '--batch_size=6',
#                  '--checkpoint_epochs=20',
#                  '--stop_after_checkpoint=0',
#                  '--threads=0',
#                  '--debug=1',
#                  '--tb_number_img=10']
# opt = parser.parse_args(training_args)
opt = parser.parse_args()
