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

# Import dataset class
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
# net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'instance', False, 'normal', 0.02, gpu_id=device)
if opt.load_pretrained == 0:
    net_g = define_G(opt.input_nc, opt.output_nc, opt.ngf, 'batch', False, 'normal', 0.02, gpu_id=device)
    net_d = define_D(opt.input_nc + opt.output_nc, opt.ndf, 'basic', gpu_id=device)
else:
    print('Loading pretrained models')
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
    checkpoint_g = torch.load(join(opt.dest_train, 'checkpoint', 'netG_model_epoch_100.pth'), map_location=torch.device(device)).to(device)
    checkpoint_d = torch.load(join(opt.dest_train, 'checkpoint', 'netD_model_epoch_100.pth'), map_location=torch.device(device)).to(device)
    net_g = torch.load_state_dict(checkpoint_g['net_g'])
    net_d = torch.load_state_dict(checkpoint_d['net_d'])

criterionGAN = GANLoss().to(device)
criterionL1 = nn.L1Loss().to(device)
criterionMSE = nn.MSELoss().to(device)

# setup optimizer
if opt.load_pretrained == 0:
    optimizer_g = optim.Adam(net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_d = optim.Adam(net_d.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
   print('Loading pretrained optimizers')
   # optimizer_g = torch.load(join(opt.dest_train, 'checkpoint', 'adam_g_epoch_100.pth'), map_location=torch.device(device))
   # optimizer_d = torch.load(join(opt.dest_train, 'checkpoint', 'adam_d_epoch_100.pth'), map_location=torch.device(device))
   optimizer_g = torch.load_state_dict(checkpoint_g['optim_g'])
   optimizer_d = torch.load_state_dict(checkpoint_d['optim_d'])
  
net_g_scheduler = get_scheduler(optimizer_g, opt)
net_d_scheduler = get_scheduler(optimizer_d, opt)

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

        print_debug(2, "Updating discriminator")

        ######################
        # (1) Update D network
        ######################

        optimizer_d.zero_grad()
        
        # train with fake
        # DTT Concatenates the real mask with generated image
        #     ** Code from original pix2pix implementation:
        #     ** fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)

        # DTT Discriminator's prediction stating if the couple of images are
        #     (real, real) o (real, false)
        #     detach() to avoid calculating gradients
        #     ** Code from original pix2pix implementation:
        #     ** pred_fake = self.netD(fake_AB.detach())
        #     ** # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = net_d.forward(fake_ab.detach())

        # DTT Calculated losses where extremely big. Debug message to see why
        print_debug(2, "Train: pred_fake's shape {}, min {} and max {}".format(
            pred_fake.shape, pred_fake.min(), pred_fake.max()
        ))

        # DTT Loss when a generated image is fed. Should classificate it as False
        #     ** Code from original pix2pix implementation:
        #     ** self.loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_d_fake = criterionGAN(pred_fake, False)

        # train with real
        # DTT Concatenates the same real mask with its corresponding real image
        #     ** Code from original pix2pix implementation:
        #     ** real_AB = torch.cat((self.real_A, self.real_B), 1)
        real_ab = torch.cat((real_a, real_b), 1)
        # DTT Discriminator's prediction. Now calculating gradients
        #     ** Code from original pix2pix implementation:
        #     ** pred_real = self.netD(real_AB)
        pred_real = net_d.forward(real_ab)
        # DTT Discriminator should predict True with a real mask + image couple
        #     ** Code from original pix2pix implementation:
        #     ** self.loss_D_real = self.criterionGAN(pred_real, True)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined D loss
        # DTT D's loss is the mean between its capacity ot detect a generated image
        #     and its capacity to detect a real image
        #     ** Code from original pix2pix implementation:
        #     ** # combine loss and calculate gradients
        #     ** self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()
       
        optimizer_d.step()

        ######################
        # (2) Update G network
        ######################

        print_debug(2, "Updating generator")

        # DTT In the pix2pix original implementation, discriminator's gradients
        #     are deactivated
        #     ** self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        optimizer_g.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = net_d.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.lamb
        loss_g = loss_g_gan + loss_g_l1
        loss_g.backward()
        optimizer_g.step()

        # DTT Let's print just some iteration messages per epoch
        #     Iterations go from 1 to ceiling(len(train_set) / batch_size)
        if iteration % max((math.ceil(len(train_set) / opt.batch_size) // opt.iter_messages), 1) == 0:
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
            epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
        
        # DTT Logging the same data for TensorBoard analysis
        if opt.tb_active:
            save_iteration_tensorboard(training_data_loader, writer_train, epoch, iteration, loss_d, loss_g, loss_g_gan, loss_g_l1,
            real_a, real_b, fake_b, batch)

        # DTT deleting GPU references just to see if the garbage collector deallocates the memory while testing is done
        # del real_a,real_b, fake_b , fake_ab , pred_fake

    # Only execute if a minimum epochs are expected
    if (opt.niter + opt.niter_decay + 1) > opt.checkpoint_epochs:
        update_learning_rate(net_g_scheduler, optimizer_g)
        update_learning_rate(net_d_scheduler, optimizer_d)

    # TEST
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = batch[0].to(device), batch[1].to(device)

        prediction = net_g(input)
        mse = criterionMSE(prediction, target)
        psnr = 10 * log10(1 / mse.item())
        avg_psnr += psnr
    # del input, target, prediction

    time_spent = time.time() - epoch_start_time
    print("===> Avg. PSNR: {:.4f} dB. Time spent in epoch {} is {:.4f}".format(avg_psnr / len(testing_data_loader), epoch, time_spent))

    if opt.tb_active:
        # DTT I log the same data for TensorBoard analysis
        print_debug(2,"  Adding scalars to tensorboard")
        writer_train.add_scalar('Metrics/Avg. PSNR', avg_psnr / len(testing_data_loader), epoch)
        writer_train.add_scalar('Metrics/D''s LR', optimizer_d.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('Metrics/G''s LR', optimizer_g.param_groups[0]['lr'], epoch)
        writer_train.add_scalar('Time spent', time_spent, epoch)

    #checkpoint
    exit = save_checkpoint(epoch, net_g, net_d, optimizer_g, optimizer_d)
    if exit:
        print("Ending training as stop_after_checkpoint is set to True")
        break

if opt.tb_active:
    writer_train.close()

train_time = time.gmtime(time.time() - start_time)
train_time = time.strftime("%H:%M:%S",train_time)

print("\nTraining ended. It took {}".format(train_time))
print("Arguments used: {}".format(opt))

