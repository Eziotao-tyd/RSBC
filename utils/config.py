# default configs 
import argparse

def add_model_group(group):
    # Model parameters
    group.add_argument('--model', type=str, default='T', help='Model to use')
    group.add_argument('--input_dim', type=int, default=467, help='Input dimension')
    group.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    group.add_argument('--num_classes', type=int, default=20, help='Number of classes')
    group.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    group.add_argument('--num_encoder_layers', type=int, default=3, help='Number of encoder layers in transformer')
    group.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')

    # Training parameters
    group.add_argument('--num_fold', type=int, default=5, help='Number of folds for cross validation')
    group.add_argument('--train_epochs', type=int, default=2000, help='Number of training epochs')
    group.add_argument('--log_every', type=int, default=500, help='Log training results every n epochs')
    group.add_argument('--lr_decay_steps', type=int, nargs='+', default=[25, 35], help='Epochs to decay learning rate')
    group.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')

def add_data_group(group):
    # Data parameters
    group.add_argument('--data_path', type=str, default='../dataset/data_2_relabelled.csv', help='Path to the training dataset')
    group.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    group.add_argument('--shuffle_dataset', type=bool, default=True, help='Whether to shuffle dataset during training')


def parse_args():
    parser = argparse.ArgumentParser()
    data_group = parser.add_argument_group(title="Data-related configuration")
    model_group = parser.add_argument_group(title="Model-related configuration")

    add_data_group(data_group)
    add_model_group(model_group)

    return parser.parse_args()