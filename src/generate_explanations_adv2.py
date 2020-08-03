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
    argparser.add_argument('--output_dir', type=str, default='../data/', help='directory to save results to')
    argparser.add_argument('--dataset', type=str, default='fmnist', help='dataset to generate explanations for')
    argparser.add_argument('--role', type=str, default='defender', help='defender or attacker', choices=['defender','adversary'])
    argparser.add_argument('--adv_dir', type=str, default='../../xai-adv/data/postndss/adv2/{}/{}/{}/{}/target_next/target_{}/', help='directory to load adv2 samples from. Format: role, dataset, target_exp_method, adv_attack_method, target_class_idx')
    argparser.add_argument('--attack_method', type=str, default='cwl2/conf_0', help='attack method', choices=['cwl2/conf_0','cwlinf/conf_0','cwl0/conf_0','bim','mim','jsma'])
    argparser.add_argument('--use_all_exp_method', help='use all explanation methods one by one', action='store_true')
    argparser.add_argument('--use_all_attack_method', help='use all attack methods one by one', action='store_true')
    argparser.add_argument('--adv2', help='input samples are adv2 examples created by attacking explanation techniques', action='store_true')
    argparser.add_argument('--target_exp_method', help='target explation method for creating adv2 samples',
                   choices=['lrp', 'guided_backprop', 'gradient', 'integrated_grad',
                            'pattern_attribution', 'grad_times_input'],
                   default='lrp')
    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    use_all_exp_method = True if args.use_all_exp_method else False
    use_all_attack_method = True if args.use_all_attack_method else False
    adv2 = True if args.adv2 else False

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
    if args.dataset in ['mnist', 'fmnist']:
        num_ch = 1
        side = 28
    elif args.dataset == 'cifar10':
        num_ch = 3
        side = 32

    print('\n\n$$$$$$ Dataset : {} $$$$$$$\n\n'.format(args.dataset))
    for exp_method in exp_methods:
        print('\n--------Computing explanations with: {}------------'.format(exp_method))
        method = getattr(ExplainingMethod, exp_method)

        # load model
        data_mean = np.array([0.0, 0.0, 0.0])
        data_std = np.array([1.0, 1.0, 1.0])

        vgg_model = torchvision.models.vgg16(pretrained=True)
        model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
        if method == ExplainingMethod.pattern_attribution:
            model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
        model = model.eval().to(device)

        for attack_method in attack_methods:
            print('\n##########Attack Method: {}#########'.format(attack_method))

            for target_class_idx in range(10):
                print('\n*************Target Class Index: {}*************'.format(target_class_idx))
                if target_class_idx in []:
                    continue
                if target_class_idx == 0:
                    adv_src_class_idx = 9
                else:
                    adv_src_class_idx = target_class_idx-1

                adv_dir = args.adv_dir.format(args.role, args.dataset, args.target_exp_method, attack_method, str(target_class_idx))
                filename = 'x_adv2.pt'

                x_train = torch.load(adv_dir + '/' + filename)
                print('Loading {} adv2 samples from {} '.format(x_train.shape[0],adv_dir))
                #x_train = (x_train*255).astype(np.uint8)
                indices = np.array([i for i in range(x_train.shape[0])])
                desired_index = random.randint(0,999)

                num_samples = indices.shape[0]
                expls = np.zeros((num_samples, side, side))

                for i,idx in enumerate(indices):
                    if i==0 or (i+1)%5 == 0:
                        print('Running for sample {}/{}'.format(i+1,num_samples))
                    #x = np_img_to_tensor(x_train[idx], data_mean, data_std, device, num_ch)
                    x = x_train[idx].to(device)
                    x_adv = x.clone().detach().requires_grad_()

                    # obtain the explanation
                    adv_expl, _, _ = get_expl(model,x_adv, method)
                    adv_expl = adv_expl.detach().cpu()
                    adv_expl_np = adv_expl.numpy()
                    adv_expl_np = adv_expl_np.reshape(224, 224)
                    im2 = Image.fromarray(adv_expl_np)
                    adv_expl2 = torchvision.transforms.ToTensor()(torchvision.transforms.Resize(side)(im2))
                    adv_expl_np2 = adv_expl2.numpy()
                    adv_expl_np2 = adv_expl_np2.reshape(side, side)

                    expls[i] = adv_expl_np2

                # store the results
                output_dir = args.output_dir + args.role + '/' + args.dataset + '/' + 'adv2/' + args.target_exp_method + '/' + attack_method + '/' + exp_method + '/from_' + str(adv_src_class_idx) + '/'

                if not os.path.exists(output_dir):
                    print('creating directory ',output_dir)
                    os.makedirs(output_dir)

                print('storing results in ',output_dir)
                np.save(output_dir+'expls.npy', expls)

                print('Done for this target_class_idx\n\n')
            print('Done for this attack method')
        print('Done for this exp_method')

    print('Process is complete. Exiting.')

if __name__ == "__main__":
    main()
