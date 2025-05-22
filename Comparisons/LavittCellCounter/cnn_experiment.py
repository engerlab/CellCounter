from fastai.vision.all import *
import os
import matplotlib.pyplot as plt
import argparse
import datetime

# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=

#Training settings
parser = argparse.ArgumentParser(description='CNN Regressor')

#Arguments
#training
parser.add_argument('--batch_size', type=int, default=32, metavar='BStrain',
                    help='input batch size (default: 32)')
parser.add_argument('--epochs', type=int, default=400, metavar='Epochs',
                    help='number of epochs to train (default: 400)')
parser.add_argument('--image_size', type=int, default=128, metavar='ImgSize',
                    help='size to resize images to (default: 128)')
parser.add_argument('--augmentations', default=True, action='store_true',
                    help='apply augmentations to data (default: True)')

#model
parser.add_argument('--pretrained', default=True, action='store_true',
                    help='load a pretrained xResNet50 model (default: True)')
parser.add_argument('--load_model_from_paper', default=True, action='store_true',
                    help='load the model as trained in the paper (default: True)')
parser.add_argument('--training', default=False, action='store_true',
                    help='run model in training mode (default: False)')
parser.add_argument('--inference', default=True, action='store_true',
                    help='run model in inference mode (default: True)')

#folders
parser.add_argument('--data_folder', type=str, default='/Users/falkolavitt/Python/CNN-regressor/data/', metavar='DF',
                    help='path to folder where the data is located (default: /working_directory/data/)')
parser.add_argument('--models_folder', type=str, default='/Users/falkolavitt/Python/CNN-regressor/models/cnn-regressor/models/', metavar='MF',
                    help='path to folder where the models are located and results are saved (default: models/cnn-regressor/models/)')

#cuda
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='If false, enables CUDA training (default: True')

#seed
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0} if not args.cuda else {}

# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=
# -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -= -=

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:19]

    model_name = 'model' + '_' + args.model_signature

    #### Set directories for saving
    model_dir = args.models_folder + 'results' + '_' + 'bs-' + str(args.batch_size) + '_' + 'epochs-' + str(args.epochs) + '_' + 'imgsize-' + str(args.image_size) + '_' + 'augmentations-' + str(args.augmentations) + '_pretrained-' + str(args.pretrained) + '/'

    if args.training:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    #### Prepare dataloader
    print('Loading Data')

    # obtain images from folder location
    path = Path(args.data_folder)
    fnames = get_image_files(path)

    # obtain labels from filenames
    ground_truth = [[113, 90, 99, 87, 96, 93, 92, 95, 96, 76, 60, 63, 10, 11, 10, 1, 1, 1],
                    [145, 175, 160, 167, 156, 120, 395, 446, 436, 107, 101, 92, 43, 28, 45, 4, 3, 6],
                    [103, 112, 72, 74, 76, 100, 62, 74, 79, 71, 83, 79, 8, 11, 10, 3, 2, 12],
                    [170, 176, 160, 139, 137, 156, 101, 98, 93, 96, 102, 87, 27, 29, 20, 7, 4, 4],
                    [165, 159, 129, 149, 164, 160, 90, 89, 103, 116, 127, 128, 18, 18, 16, 3, 8, 9],
                    [139, 140, 123, 117, 106, 121, 89, 100, 79, 91, 100, 102, 14, 22, 9, 7, 10, 7],
                    [160, 153, 144, 140, 140, 151, 149, 143, 146, 166, 168, 169, 25, 20, 20, 14, 11, 9],
                    [157, 160, 117, 138, 140, 155, 144, 118, 117, 138, 125, 108, 27, 29, 25, 6, 11, 10],
                    [169, 156, 173, 217, 171, 170, 123, 125, 109, 166, 143, 173, 32, 19, 21, 8, 6, 4],
                    [153, 156, 135, 157, 138, 163, 94, 92, 72, 119, 117, 92, 31, 31, 21, 4, 9, 8],
                    [147, 158, 143, 156, 149, 119, 118, 143, 121, 168, 175, 144, 28, 27, 18, 10, 8, 13],
                    [137, 115, 115, 126, 122, 136, 93, 105, 114, 137, 119, 142, 28, 29, 25, 8, 12, 7],
                    [228, 240, 294, 259, 243, 239, 1, 1, 0],
                    [114, 98, 115, 104, 98, 121, 99, 75, 75, 59, 76, 62, 9, 9, 14, 4, 1, 0],
                    [220, 202, 213, 176, 209, 201, 112, 127, 101, 65, 62, 64, 18, 23, 31, 5, 3, 5],
                    [145, 152, 167, 164, 167, 166, 114, 82, 74, 59, 12, 10, 13, 1, 3, 4],
                    [61, 59, 62, 60, 67, 57, 30, 19, 36, 33, 45, 32, 6, 7, 16, 16, 16, 7, 15, 15, 17, 33, 28, 28],
                    [103, 95, 87, 87, 83, 90, 44, 48, 51, 45, 63, 55, 13, 19, 17, 19, 20, 21, 6, 7, 2, 4, 2, 4],
                    [74, 72, 93, 65, 76, 87, 60, 55, 49, 59, 49, 66, 16, 11, 9, 15, 12, 14, 4, 2, 5, 7, 4, 8],
                    [8, 11, 13, 6, 10, 5, 46, 55, 53, 69, 67, 52, 8, 11, 14, 6, 19, 12, 12, 18, 11, 14, 15, 11],
                    [70, 90, 80, 96, 93, 76, 68, 63, 65, 78, 75, 59, 10, 6, 10, 8, 16, 9, 3, 2, 4, 6, 5, 4],
                    [43, 48, 46, 34, 41, 41, 12, 16, 8, 25, 24, 20, 3, 7, 12, 6, 7, 5, 4, 2, 2, 1, 2, 0],
                    [133, 130, 99, 105, 117, 110, 34, 49, 55, 48, 55, 65, 17, 22, 14, 14, 16, 13, 2, 2, 3, 8, 1, 6],
                    [29, 40, 5, 12, 9, 6],
                    [118, 102, 108, 101, 112, 113, 46, 45, 35, 47, 54, 45, 11, 16, 16, 12, 20, 17, 3, 2, 1, 4, 3, 1],
                    [144, 117, 147, 149, 125, 122, 56, 73, 52, 40, 55, 51, 21, 29, 30, 23, 19, 32, 9, 5, 3, 8, 3, 5],
                    [131, 148, 160, 155, 158, 132, 96, 110, 79, 116, 123, 108, 46, 33, 46, 40, 45, 42, 42, 26, 40, 42, 30, 36], #27
                    [157, 144, 143, 142, 159, 163, 78, 78, 70, 149, 153, 124, 53, 43, 44, 49, 45, 45, 28, 26, 22, 24, 20, 20],
                    [134, 128, 108, 121, 111, 105, 94, 85, 84, 110, 126, 116, 53, 48, 49, 51, 51, 40, 41, 46, 36, 38, 42, 37],
                    [159, 134, 163, 165, 179, 147, 94, 95, 92, 158, 149, 158, 41, 39, 39, 38, 45, 47, 36, 30, 32, 34, 36, 25],
                    [181, 136, 161, 158, 155, 130, 70, 74, 72, 128, 140, 125, 51, 42, 46, 52, 57, 48, 18, 25, 27, 20, 23, 19],
                    [137, 141, 137, 147, 158, 124, 82, 75, 80, 166, 143, 130, 28, 34, 39, 47, 39, 46, 17, 25, 21, 19, 22, 20],
                    [146, 146, 164, 125, 150, 160, 111, 110, 98, 134, 120, 129, 64, 46, 52, 60, 43, 55, 36, 34, 42, 31, 28, 36],
                    [109, 107, 110, 78, 139, 120, 40, 57, 73, 60, 65, 66, 14, 29, 31, 26, 22, 25, 8, 8, 10, 11, 16, 7],
                    [131, 114, 139, 110, 125, 123, 104, 105, 118, 83, 95, 90, 13, 19, 18, 15, 17, 12, 10, 12, 7, 7, 7, 7],
                    [114, 164, 154, 141, 129, 118, 112, 122, 119, 160, 184, 175, 30, 36, 34, 17, 33, 33, 19, 17, 8, 8, 17, 18],
                    [119, 130, 132, 135, 124, 140, 87, 107, 109, 132, 134, 100, 22, 14, 22, 19, 23, 17, 10, 4, 8, 9, 6, 4],
                    [146, 143, 127, 148, 123, 112, 81, 102, 82, 90, 106, 101, 35, 35, 28, 31, 34, 28, 9, 9, 8, 9, 11, 9]]
    get_y = lambda x: ground_truth[int(str(x).split('-')[1].split('_')[-1]) - 1][int(str(x).split('-')[2].split('.')[0]) - 1]
    # float(str(x).split('.')[-2])
    
    splitter = RandomSplitter(valid_pct=0.2, seed=42)

    # apply transformations
    item_tfms = [RatioResize(args.image_size)]
    if args.augmentations:
        batch_tfms=[*aug_transforms(mult=0, flip_vert=True, max_rotate=45, min_zoom=0, max_zoom=0, max_warp=0, p_affine=0), Normalize.from_stats(*imagenet_stats)]

    # create datablocks and dataloader
    blocks = (ImageBlock, RegressionBlock)
    block = DataBlock(blocks=blocks,
                      get_items=get_image_files,
                      get_y=get_y,
                      splitter=splitter,
                      item_tfms=item_tfms,
                      batch_tfms=batch_tfms)

    dls = block.dataloaders(path, bs=args.batch_size, num_workers=kwargs['num_workers'])

    # move dataloader to cuda
    if args.cuda:
        dls.cuda()

    #### Model
    print('Loading Model')
    # create xResNet50
    learn = Learner(dls, xresnet50(pretrained=args.pretrained, n_out=1), metrics=mae)

    # move model to cuda
    if args.cuda:
        learn.model = learn.model.cuda()

    # load pretrained model from paper
    if args.load_model_from_paper:
        learn.load(f'{os.getcwd()}/model.pth')

    # train model
    if args.training:
        print('Starting Training')
        learn.fine_tune(args.epochs, cbs=SaveModelCallback(fname=(model_dir + model_name)))

    # load test data into dataloader
    print('Loading test data')
    imgs = get_image_files(f'{args.data_folder}')

    if args.cuda:
        dl = learn.dls.test_dl(imgs, with_labels=True)
        dl.cuda()
    else:
        dl = learn.dls.test_dl(imgs, with_labels=True, num_workers=kwargs['num_workers'])

    # print performance on test set
    print('Performance: \nSum of error and Mean Absolute Error:')
    res = learn.validate(dl=dl)
    print(res)

    if args.training:
        # save performance
        f = open(f'{model_dir}{model_name}.txt', 'w+')
        f.write(str(args) + '\n' + str(res))
        f.close()

        # create boxplot of errors
        plt.boxplot(abs(res[0].view((args.batch_size)) - res[1]), labels=['CNN'])
        plt.savefig(f'{model_dir}{model_name}.png')
        plt.close()

if __name__ == "__main__":
    run(args, kwargs)