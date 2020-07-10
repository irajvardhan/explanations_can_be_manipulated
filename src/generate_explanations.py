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

def np_img_to_tensor(grayscale_img,data_mean,data_std, device):
    rgb_img = np.repeat(grayscale_img[..., np.newaxis], 3, -1)
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
    argparser.add_argument('--class_idx', type=int, default=6, help='class index to generate explanations for')
    argparser.add_argument('--start_idx', type=int, help='threshold on number of samples', default=0)
    argparser.add_argument('--end_idx', type=int, help='threshold on number of samples', default=1000)
    argparser.add_argument('--use_test_set', help='use test set instead of train set', action='store_true')
    argparser.add_argument('--adv_dir', type=str, default='../../xai-adv/data/postndss/{}/{}/target_next/target_{}/{}/', help='directory to load adv samples from. Format: role, dataset, target_class_idx, adv_attack_method')
    argparser.add_argument('--adv_src_class_idx',type=int, default=5, help='source class index from which adv sample was created')
    argparser.add_argument('--attack_method', type=str, default='cwl2/conf_0', help='attack method', choices=['cwl2/conf_0','bim','mim','jsma'])
    argparser.add_argument('--input_is_adv', help='input is adversarial samples', action='store_true')
    argparser.add_argument('--use_all_exp_method', help='use all explanation methods one by one', action='store_true')

    args = argparser.parse_args()

    # options
    device = torch.device("cuda" if args.cuda else "cpu")
    input_is_adv = True if args.input_is_adv else False
    use_test_set = True if args.use_test_set else False
    use_all_exp_method = True if args.use_all_exp_method else False

    exp_methods = []
    if use_all_exp_method:
        exp_methods = ['guided_backprop', 'integrated_grad','pattern_attribution', 'grad_times_input'] 
    else:
        exp_methods.append(args.method)

    if input_is_adv:
        target_class_idx = (args.adv_src_class_idx+1)%10
        adv_dir = args.adv_dir + str(target_class_idx) + '/' + args.attack_method + '/'
        adv_dir = args.adv_dir.format(args.role, args.dataset,str(target_class_idx), args.attack_method)

    for exp_method in exp_methods:
        print('\n--------Computing explanations with {}------------'.format(exp_method))
        method = getattr(ExplainingMethod, exp_method)

        # load model
        data_mean = np.array([0.0, 0.0, 0.0])
        data_std = np.array([1.0, 1.0, 1.0])

        vgg_model = torchvision.models.vgg16(pretrained=True)
        model = ExplainableNet(vgg_model, data_mean=data_mean, data_std=data_std, beta=1000 if args.beta_growth else None)
        if method == ExplainingMethod.pattern_attribution:
            model.load_state_dict(torch.load('../models/model_vgg16_pattern_small.pth'), strict=False)
        model = model.eval().to(device)

        if not input_is_adv:
            (x_train, y_train) ,(x_test, y_test) = keras.datasets.fashion_mnist.load_data()
            if use_test_set:
                print('Generating explanations on test set')
                x_train, y_train = x_test, y_test

            indices = np.where(y_train==args.class_idx)[0][args.start_idx:args.end_idx]
        else:

            x_train = np.load(adv_dir + '/x_adv_ar.npy')
            print('Loading {} adv samples from {} '.format(x_train.shape[0],adv_dir))
            x_train = (x_train*255).astype(np.uint8)
            indices = np.array([i for i in range(x_train.shape[0])])
            desired_index = random.randint(0,999)

        num_samples = indices.shape[0]

        # expls will store explanations for all the samples
        expls = np.zeros((num_samples,28,28))

        for i,idx in enumerate(indices):
            if (i+1)%500 == 0:
                print('Running for sample {}/{}'.format(i+1,num_samples))
            x = np_img_to_tensor(x_train[idx], data_mean, data_std, device)
            x_adv = x.clone().detach().requires_grad_()

            # obtain the explanation
            if not input_is_adv:
                org_expl, org_acc, org_idx = get_expl(model, x, method)
                org_expl = org_expl.detach().cpu()
            else:
                org_expl, org_acc, org_idx = get_expl(model, x, method, desired_index)
                org_expl = org_expl.detach().cpu()

            # convert explanation to numpy and subsequently downsize it to 28x28
            org_expl_np = org_expl.numpy()
            org_expl_np = org_expl_np.reshape(224, 224)
            im2 = Image.fromarray(org_expl_np)
            org_expl2 = torchvision.transforms.ToTensor()(torchvision.transforms.Resize(28)(im2))
            org_expl_np2 = org_expl2.numpy()
            org_expl_np2 = org_expl_np2.reshape(28,28)

            expls[i] = org_expl_np2


        # store the results
        if not input_is_adv:
            if not use_test_set:
                output_dir = args.output_dir + '/' + args.role + '/' + args.dataset + '/' + 'orig/train/' + exp_method + '/' + str(args.class_idx) + '/'
            else:
                output_dir = args.output_dir + '/' + args.role + '/' + args.dataset + '/' + 'orig/test/' + exp_method + '/' + str(args.class_idx) + '/'
        else:
            output_dir = args.output_dir + '/' + args.role + '/' + args.dataset + '/' + 'adv/' + args.attack_method + '/' + exp_method + '/from_' + str(args.adv_src_class_idx) + '/'

        if not os.path.exists(output_dir):
            print('creating directory ',output_dir)
            os.makedirs(output_dir)

        print('storing results in ',output_dir)
        np.save(output_dir+'expls.npy', expls)

        print('Done for this explanation method\n\n')

    print('Process is complete. Exiting.')

if __name__ == "__main__":
    main()
