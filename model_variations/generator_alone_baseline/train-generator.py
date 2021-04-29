# Script to train only the generator. There won't be any discriminator acting
# as a loss funcion

# Accessing files
from os.path import join

# Loading images
from torch.utils.data import DataLoader

# Debugging
import math  ## math.ceil used in debugging

# Dealing with GPUs
import torch.backends.cudnn as cudnn
import torch  ## Used in reference to torch.cuda

# Defining the networks
import torch.nn as nn  ## nn losses used
import torch.optim as optim  ## Adam used

# Training
from math import log10  ## Used in PSNR calculus
import time ## Used to calculate training times

# Import arguments
from arguments import opt

# Import dataset
from dataset import DatasetFromFolder

# Import model architecture
from network import define_G, define_D, GANLoss, get_scheduler, update_learning_rate

# Import utils
from utils import *

# Check if GPU is available
if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

device = torch.device("cuda:0" if opt.cuda else "cpu")


# INSTANTIATE DATASET CLASS AND POINT TO SOURCE DIRECTORY
train_set = DatasetFromFolder(opt.dest_train, opt.direction) # a2b is "gt" to "images"
test_set  = DatasetFromFolder(opt.dest_valid, opt.direction)  # b2a is "images" to "gt"

# INSTANTIATE DATALOADER
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=True)

# CREATING NETWORKS
if opt.load_pretrained == 0:
    net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
else:
    print('Loading pretrained model')
    net_g = torch.load(join(opt.dest_train, 'checkpoint', 'netG_model_epoch_100.pth'), map_location=torch.device(device)).to(device)
    net_g = torch.load_state_dict(checkpoint_g['net_g'])

criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
if opt.load_pretrained == 0:
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    print('Loading pretrained optimizer')
    optimizer_g = torch.load_state_dict(checkpoint_g['optim_g'])
  
net_g_scheduler = get_scheduler(optimizer_g, opt)

# SETTING-UP TENSORBOARD WRITER
train_tensorboard_dir = join(opt.dest_train, opt.log_dir)
test_tensorboard_dir = join(opt.dest_valid, opt.log_dir)
if opt.tb_active:
    writer_train = setup_tensorboard_writer(train_tensorboard_dir, model=net_g)
output_images = []
start_time = time.time()


# INITIATE TRAINING LOOP
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    print_debug(2, "Starting epoch {}".format(epoch))
    epoch_start_time = time.time()
    # TRAIN
    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        # DTT a: masks, b: satellite image
        real_a, real_b = batch[0].to(device), batch[1].to(device)
        # DTT fake_b: generated satellite image from generator
        #     ** Code from original pix2pix implementation:
        #     ** self.fake_B = self.netG(self.real_A)  # G(A)
        fake_b = net_g(real_a)

        ######################
        # (2) Update G network
        ######################

        print_debug(2, "Updating generator")

        optimizer_g.zero_grad()

        # G(A) = B
        loss_g = criterionL1(fake_b, real_b) * opt.lamb
        loss_g.backward()
        optimizer_g.step()

        # DTT Let's print just some iteration messages per epoch
        #     Iterations go from 1 to ceiling(len(train_set) / batch_size)
        if iteration % max((math.ceil(len(train_set) / opt.batch_size) // opt.iter_messages), 1) == 0:
            print("===> Epoch[{}]({}/{}): Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_g.item()))
        
        # DTT Logging the same data for TensorBoard analysis
        if opt.tb_active:
            save_iteration_tensorboard(training_data_loader, writer_train, epoch, iteration, loss_g,
            real_a, real_b, fake_b, batch)

    # Only execute if a minimum epochs are expected
    if (opt.niter + opt.niter_decay + 1) > opt.checkpoint_epochs:
        update_learning_rate(net_g_scheduler, optimizer_g)

    # TEST
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr

    time_spent = time.time() - epoch_start_time
    print("===> Avg. PSNR: {:.4f} dB. Time spent in epoch {} is {:.4f}".format(avg_psnr / len(testing_data_loader), epoch, time_spent))

    if opt.tb_active:
        # DTT I log the same data for TensorBoard analysis
        print_debug(2,"  Adding scalars to tensorboard")
        writer_train.add_scalar('Metrics/Avg. PSNR', avg_psnr / len(testing_data_loader), epoch)
        writer_train.add_scalar('Metrics/G''s LR', optimizer_g.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('Time spent', time_spent, epoch)

    #checkpoint
    exit = save_checkpoint(epoch, net_g, optimizer_g)
    if exit:
        print("Ending training as stop_after_checkpoint is set to True")
        break

if opt.tb_active:
    writer_train.close()

train_time = time.gmtime(time.time() - start_time)
train_time = time.strftime("%H:%M:%S",train_time)

print("\nTraining ended. It took {}".format(train_time))
print("Arguments used: {}".format(opt))

