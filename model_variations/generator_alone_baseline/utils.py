# Accessing the files and preparing the dataset
from os.path import join
import os
from arguments import opt

# Treating the images
from PIL import Image
import numpy as np
from matplotlib.pyplot import imshow # used??
import matplotlib.pyplot as plt

# Dealing with GPUs
import torch.backends.cudnn as cudnn
import torch # torch.cuda used

# Tensorboard
from torch.utils.tensorboard import SummaryWriter
import datetime


# Check if GPU is available
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu") # Used only once. To revise


def print_debug(level, text):
    """
    Prints a debug message only if the level of the message is lower or equal
    to the debug level set in global variable debug
    """
    # Accessing the global debug variable
    #global debug
    # The text will only be
    if level <= debug:
        print("  [DEBUG] " + text)



def denormalize_image(image_tensor):
    """
    Denormalizes an image coming from the network, usually, a generated image

    Parameters
    ----------
    images_tensor: tensor representing a PIL image
    """
    print_debug(2, "denormalize_image image tensor shape: {}".format(image_tensor.shape))
    # cpu() to avoid error "can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first."
    image_numpy = image_tensor.cpu().data.float().numpy()
    
            # La transformación inversa sería simplemente min( (x*0.5)+0.5), 1)
            # (haciendo un clipping de los valores para que no nos salgan colores raros).
            # Tensorboard creo que ya gestiona lo del clipping;
            # pero viene de nuestra cuenta hacer la "desnormalización".

    print_debug(2, "denormalize_image image_numpy shape: {}".format(image_numpy.shape))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    print_debug(2, "denormalize_image image_numpy shape: {} after transposing".format(image_numpy.shape))
    image_numpy = image_numpy.clip(0, 255)
    print_debug(2, "denormalize_image image_numpy shape: {} after clipping".format(image_numpy.shape))
    image_numpy = image_numpy.astype(np.uint8)
    print_debug(2, "denormalize_image image_numpy shape: {} after converting to uint8".format(image_numpy.shape))

    return image_numpy

def show_image(image_tensor):
    """
    Shows an image coming from the network

    Parameters
    """
    image_numpy = denormalize_image(image_tensor)
    pil_image = Image.fromarray(image_numpy)
    imshow(pil_image)    



 # Based on utils.py save_img and the last answer in
# https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly/46616645#46616645
# Plots several figures in a tile
def show_images_grid(images_tuple, nrows=1, ncols=1):
    """
    Shows several images coming from a DataLoader based on DatasetFromFolder
    in a tile

    Parameters
    ----------
    images_tuple: tuple of tensors representing images
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15,15))
    for ind,image_tensor in zip(range(len(images_tuple)), images_tuple):
        # First, denormalize image to allow it to be printable
        image_numpy = denormalize_image(image_tensor)
        image_pil = Image.fromarray(image_numpy)
        # imshow(image_pil)
        
        axeslist.ravel()[ind].imshow(image_pil, cmap=plt.jet())
        # axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.tight_layout() # optional


def print_debug(level, text):
    """
    Prints a debug message only if the level of the message is lower or equal
    to the debug level set in global variable debug
    """
    # Accessing the global debug variable
    # global debug
    # The text will only be
    if level <= opt.debug:
        print("  [DEBUG] " + text)


# Function creating an instante for a tensorboard wrapper
def setup_tensorboard_writer(tensorboard_dir, model=None):
    """
    Creates a new directory in tensorboard_dir to log data for TensorBoard.
    If a model/net is provided, it is added to the writer.

    Returns a reference to the writer
    """
    # Setting up TensorBoard writer
    # Creates a new directory to store TensorBoard data
    log_subdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=join(tensorboard_dir, log_subdir))

    if model is not None:
        # Adding the model to TensorBoard
        # Apparently, TensorBoard only accepts one model per writer
        # TODO: try to initialize the graph with a sample from the Dataset
        writer.add_graph(model, input_to_model=torch.randn([1,3,256,256]).to(device))

    return writer


# Function creating several data from an itearation in tensorboard
# Adapted to a training with only the generator
def save_iteration_tensorboard(data_loader, writer, epoch, iteration, loss_g,
                               real_a, real_b, fake_b, batch):
    tensorboard_step = len(data_loader.dataset.image_filenames) * (epoch - opt.epoch_count) + iteration
    writer.add_scalar('Loss/G', loss_g.item(), global_step=tensorboard_step)

    # DTT Decide whether saving images to tensorboard
    final_epoch = opt.niter + opt.niter_decay + 1
                                # final_epoch / opt.tb_number_img gives the number of epochs
                                # that should pass before an image is saved.
    epochs_to_pass = max(1, final_epoch // opt.tb_number_img) # at least should be 1
    save_image_to_tensorboard = ( ( (epoch % epochs_to_pass == 0)
                                # or it is the last epoch of training
                                    or (epoch == final_epoch)
                                  )
                                  # it only saves the image if it corresponds to the defined opt.tb_image
                                  and opt.tb_image in batch[2]
    )
    if save_image_to_tensorboard:
        print_debug(1, "save_iteration_tensorboard: saving {} to TensorBoard. Is in? {}. Batch: {}".format(opt.tb_image, opt.tb_image in batch[2], batch[2]))
        
        batch_index = batch[2].index(opt.tb_image)
        # DTT Write images to TensorBoard at the end of each epoch
        writer.add_image(str(epoch)+'/1 Mask', real_a[batch_index], epoch)
        writer.add_image(str(epoch)+'/4 Denormalized generated satellite image', denormalize_image(fake_b[batch_index]), epoch, dataformats='HWC')
    elif opt.tb_image in batch[2]:
        print_debug(2, "save_iteration_tensorboard: {} in batch, but won't save it. epochs_to_pass: {}. Batch: {}".format(
            opt.tb_image, epochs_to_pass, batch[2]))
    else:
        print_debug(2, "save_iteration_tensorboard: won't save any image ({})".format(batch[2]))

    # TODO: use psutil.virtual_memory() explained in 
    # https://stackoverflow.com/questions/276052/how-to-get-current-cpu-and-ram-usage-in-python


# Function saving a model checkpoint after several epochs.
def save_checkpoint(epoch, net_g, optimizer_g):
    """
    Saves the discriminator and generator.
    It returns a boolean stating whether training should stop or not
    """
    print_debug(2, "    save_checkpoint")
    if epoch % opt.checkpoint_epochs == 0:
        print_debug(1, "    Saving checkpoint at epoch {}".format(epoch))
        checkpoint_dir = join(opt.dest_train, 'checkpoint')
        os.makedirs(name = checkpoint_dir, exist_ok=True)
        net_g_model_out_path = os.path.join(checkpoint_dir, "netG_model_epoch_{}.pth".format(epoch))
        
        # Let's get rid of saving path dependencies with state_dict
        # torch.save(net_g, net_g_model_out_path)
        torch.save({'net_g': net_g.state_dict(),
                    'optim_g': optimizer_g.state_dict()},
                   net_g_model_out_path)
        print("Checkpoint for epoch {} saved".format(epoch))

        if opt.stop_after_checkpoint == 1:
            return True
        else:
            return False
    else:
        print_debug(2, "    No checkpoint saved")
        return False
