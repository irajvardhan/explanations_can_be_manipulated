import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="4"

import argparse
import torch
import torchvision
import torch.nn.functional as F
import numpy as np

from nn.enums import ExplainingMethod
from nn.networks import ExplainableNet
from nn.utils import get_expl, plot_overview, clamp, load_image, make_dir

import keras
from keras import datasets
from torchvision import datasets, transforms
from PIL import Image
import random
import time

def get_beta(i, num_iter):
    """
    Helper method for beta growth
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)

def np_img_to_tensor(input_img,data_mean,data_std, device, num_ch=1):
    if num_ch == 1:
        rgb_img = np.repeat(input_img[..., np.newaxis], 3, -1)
    else:
        rgb_img = input_img
    im = Image.fromarray(rgb_img)
    x = torchvision.transforms.Normalize(mean=data_mean, std=data_std)(torchvision.transforms.ToTensor()(torchvision.transforms.Resize(224)(im)))
    x = x.unsqueeze(0).to(device)
    return x


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--cuda', help='enable GPU mode', action='store_true')
    argparser.add_argument('--beta_growth', help='enable beta growth', action='store_true')
    argparser.add_argument('--sample_threshold', type=int, help='threshold on number of samples', default=1000)
    argparser.add_argument('--method', help='algorithm for expls',
                       choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                                'pattern_attribution', 'grad_times_input'],
                       default='lrp')
    argparser.add_argument('--output_dir', type=str, default='../../xai-adv/data/postndss/', help='directory to save results to')
    argparser.add_argument('--dataset', type=str, default='fmnist', help='dataset to generate explanations for')
    argparser.add_argument('--role', type=str, default='defender', help='defender or attacker', choices=['defender','adversary'])
    argparser.add_argument('--adv_dir', type=str, default='../../xai-adv/data/postndss/{}/{}/target_next/target_{}/{}/', help='directory to load adv samples from. Format: role, dataset, target_class_idx, adv_attack_method')
    argparser.add_argument('--attack_method', type=str, default='cwl2/conf_0', help='attack method', choices=['cwl2/conf_0','cwlinf/conf_0','cwl0/conf_0','bim','mim','jsma'])
    argparser.add_argument('--use_all_exp_method', help='use all explanation methods one by one', action='store_true')
    argparser.add_argument('--use_all_attack_method', help='use all attack methods one by one', action='store_true')
    argparser.add_argument('--prefactors', nargs=2, default=[1e11, 1e6], type=float,
                           help='prefactors of losses (diff expls, class loss)')
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    use_all_exp_method = True if args.use_all_exp_method else False
    use_all_attack_method = True if args.use_all_attack_method else False

    exp_methods = []
    if use_all_exp_method:
        exp_methods = ['lrp','guided_backprop', 'integrated_grad','pattern_attribution', 'grad_times_input']
    else:
        exp_methods.append(args.method)
    attack_methods = []
    if use_all_attack_method:
        attack_methods = ['cwl2/conf_0', 'cwlinf/conf_0', 'cwl0/conf_0', 'bim', 'mim', 'jsma']
    else:
        attack_methods.append(args.attack_method)

    # expls will store explanations for all the samples
    if args.dataset == 'fmnist':
        num_ch = 1
        side = 28

        (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.fashion_mnist.load_data()

    elif args.dataset == 'cifar10':
        num_ch = 3
        side = 32

        (x_train_orig, y_train_orig), (x_test_orig, y_test_orig) = keras.datasets.cifar10.load_data()

    print('\n\n$$$$$$ Dataset : {} $$$$$$$\n\n'.format(args.dataset))

    # load model
    data_mean = np.array([0.0, 0.0, 0.0])
    data_std = np.array([1.0, 1.0, 1.0])

    for target_exp_method in exp_methods:
        print('\n-------- Targeting explanation method: {}------------'.format(target_exp_method))

        # Set hyperparameters
        if target_exp_method == 'lrp':
            beta_growth = False
            num_iter = 1500
            lr = 2 * (10**(-4))
        elif target_exp_method == 'guided_backprop':
            beta_growth = True
            num_iter = 1500
            lr = 1 * (10**(-3))
        elif target_exp_method == 'integrated_grad':
            beta_growth = True
            num_iter = 500
            lr = 5 * (10**(-3))
        elif target_exp_method == 'pattern_attribution':
            beta_growth = True
            num_iter = 1500
            lr = 2 * (10**(-3))
        elif target_exp_method == 'grad_times_input':
            beta_growth = True
            num_iter = 1500
            lr = 1 * (10**(-3))

        method = getattr(ExplainingMethod, target_exp_method)

        vgg_model = torchvision.models.vgg16(pretrained=True)
        model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
        if method == ExplainingMethod.pattern_attribution:
            model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
        model = model.eval().to(device)

        for attack_method in attack_methods:
            print('\n########## Attack Method: {}#########'.format(attack_method))
            for target_class_idx in range(2, 10):
                print('\n************* Target Class Index: {} *************'.format(target_class_idx))

                ## Obtain the target explanation, which will be an explanation of a normal example from the TARGET class
                idx = np.where(y_train_orig == target_class_idx)[0][0]
                grayscale_img_src = x_train_orig[idx]
                x_target = np_img_to_tensor(grayscale_img_src,data_mean,data_std, device, num_ch)
                target_expl, _, _ = get_expl(model, x_target, method)
                target_expl = target_expl.detach()

                adv_dir = args.adv_dir.format(args.role, args.dataset,str(target_class_idx), attack_method)
                x_train = np.load(adv_dir + '/x_adv_ar.npy')
                print('Loading {} adv samples from {} '.format(x_train.shape[0],adv_dir))
                x_train = x_train[0:10]
                print('Retaining {} adv samples'.format(x_train.shape[0]))
                x_train = (x_train*255).astype(np.uint8)
                indices = np.array([i for i in range(x_train.shape[0])])
                num_samples = indices.shape[0]

                x_adv2 = np.zeros((num_samples, side, side))
                succ_on_f_list = []
                time_taken_list = []
                distortions = [] # from each x_adv1 to x_adv2
                for i,idx in enumerate(indices):
                    if (i+1)%1 == 0:
                        print('Running for sample {}/{}'.format(i+1,num_samples))
                    x = np_img_to_tensor(x_train[idx], data_mean, data_std, device, num_ch)
                    x_adv = x.clone().detach().requires_grad_()

                    # Find the original class of the adversarial example (x_adv_1)
                    x_pred = model(x)
                    x_pred_class = int(torch.argmax(x_pred,axis=1))

                    # obtain the explanation
                    org_expl, org_acc, org_idx = get_expl(model, x, method)
                    org_expl = org_expl.detach().cpu()

                    # Perform the whitebox attack
                    optimizer = torch.optim.Adam([x_adv], lr=lr)
                    timestart = time.time()
                    for iter in range(num_iter):
                        if beta_growth:
                            model.change_beta(get_beta(iter, num_iter))

                        optimizer.zero_grad()

                        # calculate loss
                        adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method, desired_index=org_idx)
                        loss_expl = F.mse_loss(adv_expl, target_expl)
                        loss_output = F.mse_loss(adv_acc, org_acc.detach())
                        total_loss = args.prefactors[0]*loss_expl + args.prefactors[1]*loss_output

                        # update adversarial example
                        total_loss.backward()
                        optimizer.step()

                        # clamp adversarial example
                        # Note: x_adv.data returns tensor which shares data with x_adv but requires
                        #       no gradient. Since we do not want to differentiate the clamping,
                        #       this is what we need
                        x_adv.data = clamp(x_adv.data, data_mean, data_std)

                        # print 1st and last iteration, and every 250th iteration in between
                        if (iter==0) or (iter+1)%250 == 0 or (iter == num_iter-1):
                            print("Iteration {}: Total Loss: {}, Expl Loss: {}, Output Loss: {}".format(iter, total_loss.item(), loss_expl.item(), loss_output.item()))

                    timeend = time.time()

                    time_taken = timeend-timestart
                    print('Time taken: {} seconds'.format(time_taken))
                    time_taken_list.append(time_taken)

                    # test with original model (with relu activations)
                    model.change_beta(None)

                    # find the predicted class of x_adv2
                    x_adv_2_pred = model(x_adv)
                    x_adv_2_pred_class = int(torch.argmax(x_adv_2_pred,axis=1))

                    # check if the original class was retained
                    succ_on_f = x_adv_2_pred_class == x_pred_class
                    succ_on_f_list.append(succ_on_f)

                    ## We only need x_adv_2 for now so commenting below line
                    #adv_expl, adv_acc, class_idx = get_expl(model, x_adv, method)

                    # get the numpy form of x_adv2
                    x_adv2_big = x_adv.detach().cpu().numpy()
                    im2 = Image.fromarray(x_adv2_big[0][0])
                    x_adv2_small = torchvision.transforms.ToTensor()(torchvision.transforms.Resize(side)(im2))
                    x_adv2_small_np = x_adv2_small.numpy()
                    x_adv2_small_np = x_adv2_small_np.reshape(side, side)

                    x_adv2[i] = x_adv2_small_np

                    # compute distortion
                    x_src = x_train[idx] / 255.
                    distortion = np.sum((x_adv2_small_np - x_src)**2)**.5
                    distortions.append(distortion)
                    print('Distortion produced: ', distortion)

                succ_on_f_list = np.array(succ_on_f_list)
                mean_time_taken = np.array([np.mean(time_taken_list)])
                mean_distortion = np.array([np.mean(distortions)])
                # store the results
                output_dir = args.output_dir + 'adv2/' + '/' + args.role + '/' + args.dataset + '/' + target_exp_method + '/' + attack_method + '/' +  'target_next' + '/' + '/target_' + str(target_class_idx) + '/'

                if not os.path.exists(output_dir):
                    print('creating directory ',output_dir)
                    os.makedirs(output_dir)

                print('storing results in ',output_dir)
                np.save(output_dir+'x_adv2.npy', x_adv2)
                np.save(output_dir+'succ_on_f.npy', succ_on_f_list)
                np.save(output_dir+'timetaken.npy', mean_time_taken)
                np.save(output_dir+'mean_distortion.npy', mean_distortion)

                print('Done for this target_class_idx \n\n')
            print('Done for this attack method')
        print('Done for this target_exp_method')

    print('Process is complete. Exiting.')

if __name__ == "__main__":
    main()
