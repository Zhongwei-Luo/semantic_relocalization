"""Common arguments for train and evaluation for RPMNet"""
import argparse


def rpmnet_arguments():
    """Arguments used for both training and testing"""
    parser = argparse.ArgumentParser(add_help=False)

    # Logging
    parser.add_argument('--logdir', default='../logs', type=str,
                        help='Directory to store logs, summaries, checkpoints.')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dev', action='store_true', help='If true, will ignore logdir and log to ../logdev instead')
    parser.add_argument('--name', type=str, help='Prefix to add to logging directory')
    parser.add_argument('--debug', default = False, help='If set, will enable autograd anomaly detection')
    # settings for input data_loader
    parser.add_argument('-i', '--dataset_path',
                        default='/root/RPMNet/datasets/modelnet40_ply_hdf5_2048',
                        type=str, metavar='PATH',
                        help='path to the processed dataset. Default: ../datasets/modelnet40_ply_hdf5_2048')
    parser.add_argument('--dataset_type', default='modelnet_hdf',
                        choices=['modelnet_hdf', 'bunny', 'armadillo', 'buddha', 'dragon'],
                        metavar='DATASET', help='dataset type (default: modelnet_hdf)')
    parser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--noise_type', default='crop', choices=['clean', 'jitter', 'crop'],
                        help='Types of perturbation to consider')
    parser.add_argument('--rot_mag', default=0.0, type=float,
                        metavar='T', help='Maximum magnitude of rotation perturbation (in degrees)')
    parser.add_argument('--trans_mag', default=0.1, type=float,
                        metavar='T', help='Maximum magnitude of translation perturbation')
    parser.add_argument('--partial', default=[1, 1], nargs='+', type=float,
                        help='Approximate proportion of points to keep for partial overlap (Set to 1.0 to disable)')
    # Model
    parser.add_argument('--method', type=str, default='rpmnet', choices=['rpmnet'],
                        help='Model to use. Note: Only rpmnet is supported for training.'
                             '\'eye\' denotes identity (no registration), \'gt\' denotes groundtruth transforms')
    # PointNet settings
    parser.add_argument('--radius', type=float, default=0.3, help='Neighborhood radius for computing pointnet features')
    parser.add_argument('--num_neighbors', type=int, default=64, metavar='N', help='Max num of neighbors to use')
    # RPMNet settings
    parser.add_argument('--features', type=str, choices=['ppf', 'dxyz', 'xyz'], default=[ 'ppf', 'dxyz', 'xyz'],
                        nargs='+', help='Which features to use. Default: all')
    parser.add_argument('--feat_dim', type=int, default=96,
                        help='Feature dimension (to compute distances on). Other numbers will be scaled accordingly')
    parser.add_argument('--num_classes', type=int, default=40,
                        help='number of classes')

    parser.add_argument('--loss_type', type=str, choices=['mse', 'mae'], default='mae',
                        help='Loss to be optimized')
    parser.add_argument('--wt_inliers', type=float, default=1e-2, help='Weight to encourage inliers')
    # Training parameters
    parser.add_argument('--train_batch_size', default=4, type=int, metavar='N',
                        help='training mini-batch size (default 8)')
    parser.add_argument('-b', '--val_batch_size', default=24, type=int, metavar='N',
                        help='mini-batch size during validation or testing (default: 16)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Pretrained network to load from. Optional for train, required for inference.')
    parser.add_argument('--gpu', default=0, type=int, metavar='DEVICE',
                        help='GPU to use, ignored if no GPU is present. Set to negative to use cpu')
    parser.add_argument('--method_type', default='voxnet', type=str, metavar='DEVICE',
                        help='type of model {rpmnet, pointnet, voxnet, dgcnn, point2}')
    parser.add_argument('--task', default='retrival', type=str, metavar='DEVICE',
                        help='type of model {retrival, cls, segmentation}')
    parser.add_argument('--object', default=8, type=int, metavar='DEVICE',
                        help='type of model {retrival, cls, segmentation}')
    parser.add_argument('--loss', default="sraa", type=str, metavar='DEVICE',
                        help='type of model {retrival, cls, segmentation}')
    parser.add_argument('--train_data_txt', default="./data_loader/train.txt", type=str, metavar='DEVICE',
                        help='type of model {retrival, cls, segmentation}')
    parser.add_argument('--validation_data_txt', default="./data_loader/validation.txt", type=str, metavar='DEVICE',
                        help='type of model {retrival, cls, segmentation}')
    return parser


def rpmnet_train_arguments():
    """Used only for training"""
    parser = argparse.ArgumentParser(parents=[rpmnet_arguments()])

    parser.add_argument('--train_categoryfile', type=str, metavar='PATH', default='/root/RPMNet/src/data_loader/modelnet40_all.txt',
                        help='path to the categories to be trained')  # eg. './dataset/modelnet40_half1.txt'
    parser.add_argument('--val_categoryfile', type=str, metavar='PATH', default='/root/RPMNet/src/data_loader/modelnet40_all.txt',
                        help='path to the categories to be val')  # eg. './sampledata/modelnet40_half1.txt'
    # Training parameters
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate during training')
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--summary_every', default=2, type=int, metavar='N',
                        help='Frequency of saving summary (number of steps if positive, number of epochs if negative)')
    parser.add_argument('--validate_every', default=-1, type=int, metavar='N',
                        help='Frequency of evaluation (number of steps if positive, number of epochs if negative).'
                             'Also saves checkpoints at the same interval')
    parser.add_argument('--num_workers', default=30, type=int,
                        help='Number of workers for data_loader loader (default: 4).')
    parser.add_argument('--num_train_reg_iter', type=int, default=1,
                        help='Number of outer iterations used for registration (only during training)')
    parser.add_argument('--train_process', type=str, default= "train",  #"feature-only",
                        help='to seperate the network one ny one')
    parser.description = 'Train RPMNet'
    return parser


def rpmnet_eval_arguments():
    """Used during evaluation"""
    parser = argparse.ArgumentParser(parents=[rpmnet_arguments()])

    # settings for input data_loader
    parser.add_argument('--test_category_file', type=str, metavar='PATH', default='./data_loader/modelnet40_all.txt',
                        help='path to the categories to be val')
    # Provided transforms
    parser.add_argument('--transform_file', type=str,
                        help='If provided, will use transforms from this provided pickle file')
    # Save out evaluation data_loader for further analysis
    parser.add_argument('--eval_save_path', type=str, default='../eval_results',
                        help='Output data_loader to save evaluation results')
    parser.add_argument('--num_workers', default=30, type=int,
                        help='Number of workers for data_loader loader (default: 4).')

    parser.description = 'RPMNet evaluation'
    return parser
