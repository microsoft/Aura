import argparse

from source.noise_type_classification.train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data\\synthetic\\tfrecords')
    parser.add_argument('--validation_dir', default='../synthetic/tfrecords')
    parser.add_argument('--pretrained_extractor', default=None)
    parser.add_argument('--experiment_name', default='audio-tag-vggish')
    parser.add_argument('--model_name', default='VGGishClassifier')
    parser.add_argument('--num_frames', type=int, default=1000)
    parser.add_argument('--num_labels', type=int, default=526)
    parser.add_argument('--num_mel', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0, help='parameter controlling focal loss')
    parser.add_argument('--alpha', type=float, default=0.2, help='parameter controlling label smoothing')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_dir', default='./outputs')
    parser.add_argument('--path_model_tf', default=None)
    parser.add_argument('--frac_samples', type=float, default=None)
    parser.add_argument('--validation_steps', type=int)
    parser.add_argument('--balancing', default=False, action='store_true')
    parser.add_argument('--output_bias', default=False, action='store_true')
    parser.add_argument('--tags', default='../../../../data/synthetic/tags/noisyspeech_full_tags.csv')

    args = parser.parse_args()
    train(args)
