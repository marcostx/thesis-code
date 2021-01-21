import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-m', '--model_name', default='lstm_late', help='Model name')
    argparser.add_argument('-at', '--if_att', default=False, help='if use attention model', type=str2bool)
    argparser.add_argument('-d', '--dataset_name', default="mediaeval", help='Dataset name')
    argparser.add_argument('-sp', '--split_file', default="kfold.json", help='json file with kfold list videos')
    argparser.add_argument('-e', '--n_epochs', default=20, help='member of epochs', type=int)
    argparser.add_argument('-b', '--batch_size', default=20, help='Training batch size', type=int)
    argparser.add_argument('-k', '--k', default=5, help='Number of folds', type=int)
    argparser.add_argument('-nf','--n_frames', default=20, help='Number of frames', type=int)
    argparser.add_argument('-vf','--visual_feature', default=True, help='feature extrator is visual or not')
    argparser.add_argument('-ao','--att_type', default="soft_att", help='attention type : model-based, uniform, etc', type=str)
    argparser.add_argument('-ft','--feature_dim', default=1536, help='Feature dimension', type=int)
    argparser.add_argument('-fm','--feature_model', default="vgg", help='Feature name', type=str)
    argparser.add_argument('-o', '--opt_name', default='Adam', help='Optimizer name')
    argparser.add_argument('-fu', '--fusion_method', default='clusters', help='fusion method : (None, early, clusters)', type=str)
    argparser.add_argument('-ra', '--use_raw', default=False, help='use raw frames as inpu or not')
    argparser.add_argument('-qa', '--qualitative', default=False, help='bool to make qualitative analysis or not')
    argparser.add_argument('-im', '--is_mediaeval', default=True, help='bool to eval in mediaeval or not')

    argparser.add_argument('-ms', '--hidden_size', default=256, help='hidden state size',type=int)
    argparser.add_argument('-as', '--att_rnn_size', default=32, help='Size of LSTM intersnal state (for attention model)',type=int)
    argparser.add_argument('-lr','--lr', default=0.001, help='Learning rate', type=float)
    argparser.add_argument('-wd','--wd', default=0.005, help='Weigth decay', type=float)

    argparser.add_argument('-drop', '--dropout_prob', required=False, help='dropout_prob', default=0.5, type=float)
    argparser.add_argument('-tm', '--temporal_method', required=False, help='e.g. uniform weigthing, last segment, soft attention, tagm, mfn, ', default=0.0, type=float)
    argparser.add_argument('-p', '--pretrained', type=str2bool, nargs='?', const=True, default=False, help="Pretrained")
    argparser.add_argument('-c', '--checkpoint_path', required=False, default="", help="Checkpoint path")
    argparser.add_argument('-dp','--datapath', default='/home/datasets', help='Directory containing data sequences', type=str)

    args = argparser.parse_args()
    print(args)
    return args


def get_crosskfold_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument('-m', '--model_name', required=True, help='Model name')
    argparser.add_argument('-ld', '--dataset_list', required=True, help='Train dataset list by ,')
    argparser.add_argument('-e', '--n_epochs', required=True, help='Number of epochs', type=int)
    argparser.add_argument('-o', '--opt_name', required=True, help='Optimizer name')
    argparser.add_argument('-lr', '--lr', required=True, help='Learning rate', type=float)
    argparser.add_argument('-wd', '--wd', required=True, help='Weigth decay', type=float)

    argparser.add_argument('-drop', '--dropout_prob', required=False, help='dropout_prob', default=0.0, type=float)
    argparser.add_argument('-p', '--pretrained', type=str2bool, nargs='?', const=True, default=False, help="Pretrained")
    argparser.add_argument('-c', '--checkpoint_path', required=False, default="", help="Checkpoint path")
    argparser.add_argument('-da', '--data_augmentation', required=False, help='Data augmentation (c --> Classic, p --> pytorch, s --> synthetic) separated by comma ,', default="")


    args = argparser.parse_args()
    config_files = {
        "model_name": args.model_name,
        "dataset_list": args.dataset_list.split(','),
        "n_epochs": args.n_epochs,
        "opt_name": args.opt_name,
        "lr": args.lr,
        "wd": args.wd,
        "dropout_prob": args.dropout_prob,
        "pretrained": args.pretrained,
        "checkpoint_path": args.checkpoint_path,
        "data_augmentation": args.data_augmentation.split(","),
    }

    print(config_files)
    return config_files
