import argparse

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', action='store', help='what kind of pretrained architecture to use', default='vgg16')
    parser.add_argument('--lr', action='store', help='which learning rate to start with', type=float, default=0.01)
    parser.add_argument('--hidden_layers', action='store', help='hidden layers', type=int, default='1000')
    parser.add_argument('--epochs', action='store', help='# of epochs for the training', type=int, default=4)
    parser.add_argument('--device', action='store', help='use gpu to train model', default = 'cuda')
    parser.add_argument('--top_k', action='store', help='choose the top K results', type=int, default=5)
    parser.add_argument('--category_names', action='store', help='choose the file that maps classes to names', default = 'cat_to_name.json')
    parser.add_argument('--gpu', action='store', help='use gpu for inference', default = 'gpu')

    args = parser.parse_args()

    arch = args.arch
    lr = args.lr
    hidden_layers = args.hidden_layers
    epochs = args.epochs
    device = args.device
    top_k = args.top_k
    category_names = args.category_names
    gpu = args.gpu


    return(arch, lr, hidden_layers, epochs, device, top_k, category_names, gpu)
