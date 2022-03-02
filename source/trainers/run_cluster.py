import argparse

from source.clusters.train_cluster import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data\\tfrecords',
                        help='Path to the tfrecords where is the data that isd used to learn cluster centers')
    parser.add_argument('--experiment_name', default='audio-tag-cluster')
    parser.add_argument('--model_name', default='VGGishClassifier', help='Classifier for which feature space is clustered')
    parser.add_argument('--clustering_name', default='MahaKmeans', help='Clustering method. Needs to be KMeans, '
                                                                        'KMeansPlusPlus' or 'MahaKmeans')
    parser.add_argument('--num_frames', type=int, default=1000)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--num_labels', type=int, default=526)
    parser.add_argument('--num_mel', type=int, default=64)
    parser.add_argument('--input_length', type=float, default=9.99)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--num_clusters', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--labels', action='store_true', help='Indicate whether the cluster reports the average label')
    parser.add_argument('--snr', action='store_true',
                        help='Indicate whether the cluster reports the average snr')
    parser.add_argument('--save_dir', default='../../../../outputs/',
                        help='Path to directory to save a npz file that contained cluster center coordinates expressed in'
                             'unit of the model_name feature space')
    parser.add_argument('--path_model_tf', default="C:\\Users\\t-xgitiaux\\OneDrive - Microsoft\\saved_models/vggish_classifier/run_202107252116_236/",
                        help='Path to the checkpoints corresponding to the classifier whose features maos are clustered')
    parser.add_argument('--tags_file', default='../../../../data/synthetic/tags/noisyspeech_full_tags.csv',
                        help='Path to the csv ontology that maps label id to label name')

    args = parser.parse_args()
    train(args)
