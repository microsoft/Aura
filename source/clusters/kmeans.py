import tensorflow as tf
import random

from tensorflow.keras import Model


class KMeans(Model):
    """
    Batch implementation of kmeans using tensorflow
    """

    def __init__(self, n_clusters, embed_dim, n_labels=526):
        super().__init__()

        self.centroids = tf.Variable(tf.random.normal((n_clusters, embed_dim)), trainable=False)
        self.N = tf.Variable(tf.zeros(n_clusters), trainable=False)
        self.center = tf.Variable(tf.zeros(embed_dim), trainable=False)
        self.dispersion = tf.Variable(tf.zeros(n_clusters), trainable=False)
        self.dispersion_labels = tf.Variable(tf.zeros(n_clusters), trainable=False)

        self.running_centroids = tf.Variable(tf.zeros((n_clusters, embed_dim)), trainable=False)
        self.running_labels = tf.Variable(tf.zeros((n_clusters, n_labels)), trainable=False)
        self.running_N = tf.Variable(tf.zeros(n_clusters), trainable=False)
        self.running_dispersion = tf.Variable(tf.zeros(n_clusters), trainable=False)
        self.running_dispersion_labels = tf.Variable(tf.zeros(n_clusters), trainable=False)
        self.running_center = tf.Variable(tf.zeros(embed_dim), trainable=False)

        self.cluster_labels = tf.Variable(tf.zeros((n_clusters, n_labels), dtype=tf.float32), trainable=False)

        self.n_clusters = n_clusters
        self.embed_dim = embed_dim
        self.n_labels = n_labels

    def assign(self, points):
        """

        :param training:
        :param labels:
        :param eps:
        :param points:
        :return:
        """
        centroids = self.centroids

        distance = self.compute_similarity(points, centroids)
        centroid_assignment = tf.argmin(distance, 1)

        return centroid_assignment

    def compute_distance(self, points, centroid_assignment):
        """

        :param centroid_assignment:
        :return:
        """
        dist = tf.math.square(tf.nn.embedding_lookup(self.centroids, centroid_assignment) - points)
        dist = tf.reduce_sum(dist, axis=-1)

        dispersion = tf.math.unsorted_segment_sum(dist, centroid_assignment, self.n_clusters)

        self.running_dispersion.assign_add(dispersion)
        self.running_center.assign_add(tf.reduce_sum(points, axis=0))

        return dist

    def compute_distance_labels(self, labels, centroid_assignment, eps=10**-8):
        """

        :param centroid_assignment:
        :return:
        """
        labels = tf.cast(labels, dtype=tf.float32)
        centroid_labels = tf.nn.embedding_lookup(self.running_labels, centroid_assignment)

        norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(labels), -1))
        norm1 = tf.math.sqrt(tf.reduce_sum(tf.math.square(centroid_labels), -1))

        dist = tf.reduce_sum(centroid_labels * labels, -1) / (norm * norm1 + eps)
        dist = 1 - dist

        dispersion = tf.math.unsorted_segment_sum(dist, centroid_assignment, self.n_clusters)

        self.running_dispersion_labels.assign_add(dispersion)

        return dist

    def compute_similarity(self, x, c):
        """
        compute euclidean distance between x and c along the second dimension
        :param x: (B, dim)
        :param c: (ncluster, dim)
        :return: (B, nclusters)
        """
        x_exp = tf.expand_dims(x, 1)
        c_exp = tf.expand_dims(c, 0)

        dist = tf.reduce_sum(tf.math.square(x_exp - c_exp), -1)

        return dist

    def update(self, points, centroid_assignment, labels, eps=10 ** (-8)):
        """
        :param eps:
        :param labels:
        :param centroid_assignment:
        :param points:
        :return:
        """
        centroids = self.centroids
        k = centroids.shape[0]

        new_sum = tf.math.unsorted_segment_sum(points, centroid_assignment, k)
        new_n = tf.math.unsorted_segment_sum(tf.ones(tf.shape(points)[0]), centroid_assignment, k)

        new_labels = tf.math.unsorted_segment_sum(labels, centroid_assignment, k)
        new_labels = tf.cast(new_labels, dtype=tf.float32)

        new_sum = self.running_centroids + new_sum
        new_n = self.running_N + new_n

        new_labels = self.running_labels + new_labels
        self.running_labels.assign(new_labels)

        self.running_centroids.assign(new_sum)
        self.running_N.assign(new_n)

    def reset_centroids(self, eps=10**-8):

        """
        Reset running counts after assigning them to state variables
        :return: dist
        """
        n = tf.expand_dims(self.running_N, 1) + eps
        self.centroids.assign(self.running_centroids / n)
        self.cluster_labels.assign(self.running_labels / n)
        self.N.assign(self.running_N)
        self.center.assign(self.running_center)
        self.dispersion.assign(self.running_dispersion)
        self.dispersion_labels.assign(self.running_dispersion_labels)

        self.running_N.assign(tf.zeros(self.n_clusters))
        self.running_centroids.assign(tf.zeros((self.n_clusters, self.embed_dim)))
        self.running_labels.assign(tf.zeros((self.n_clusters, self.n_labels)))
        self.running_center.assign(tf.Variable(tf.zeros(self.embed_dim), trainable=False))
        self.running_dispersion.assign(tf.Variable(tf.zeros(self.n_clusters), trainable=False))
        self.running_dispersion_labels.assign(tf.Variable(tf.zeros(self.n_clusters), trainable=False))

    def compute_calinski_harabasz(self):
        """
        Compute CH index as ratio of between dispersion to within dispersion:
        http://datamining.rutgers.edu/publication/internalmeasures.pdf
        :return:
        """
        center = tf.expand_dims(self.center, 0)
        between_dispersion = tf.reduce_sum(tf.math.square(center - self.centroids), 1)
        between_dispersion *= self.N / (self.n_clusters - 1)
        between_dispersion = tf.reduce_sum(between_dispersion)

        within_dispersion = tf.reduce_sum(self.dispersion) / (tf.reduce_sum(self.N) - self.n_clusters)

        return between_dispersion / within_dispersion

    def compute_davies_bouldin(self, eps=10**-8):
        """
        Compute the davies bouldin index to measure separation between cluster:
        http://datamining.rutgers.edu/publication/internalmeasures.pdf
        The distance is the euclidean distance
        :return:
        """
        c = tf.expand_dims(self.centroids, 0)
        c1 = tf.expand_dims(self.centroids, 1)
        centroid_distance = tf.reduce_sum(tf.math.square(c - c1), axis=-1)

        dispersion = tf.expand_dims(self.dispersion / (self.N + eps), 0)
        dispersion1 = tf.expand_dims(self.dispersion / (self.N + eps), 1)

        r_index = (dispersion + dispersion1) / (centroid_distance + eps)
        r_index = tf.linalg.set_diag(r_index, tf.zeros(self.n_clusters))

        db_index = tf.reduce_sum(tf.reduce_max(r_index, 1)) / self.n_clusters

        return db_index

    def compute_davies_bouldin_labels(self, eps=10**-8):
        """
        Compute the davies bouldin index to measure separation between cluster:
        http://datamining.rutgers.edu/publication/internalmeasures.pdf
        The distance is based on the labels of each vector
        :return:
        """
        c = tf.expand_dims(self.cluster_labels, 0)
        c1 = tf.expand_dims(self.cluster_labels, 1)

        norm_c = tf.math.sqrt(tf.reduce_sum(tf.math.square(c), -1))
        norm_c1 = tf.math.sqrt(tf.reduce_sum(tf.math.square(c1), -1))

        centroid_distance = tf.reduce_sum(c * c1, axis=-1)
        centroid_distance = 1 - centroid_distance / (norm_c * norm_c1 + eps)

        dispersion = tf.expand_dims(self.dispersion_labels / (self.N + eps), 0)
        dispersion1 = tf.expand_dims(self.dispersion_labels / (self.N + eps), 1)

        r_index = (dispersion + dispersion1) / (centroid_distance + eps)
        r_index = tf.linalg.set_diag(r_index, tf.zeros(self.n_clusters))

        db_index = tf.reduce_sum(tf.reduce_max(r_index, 1)) / self.n_clusters

        return db_index


class KMeansPlusPlus(KMeans):
    """
    Batch implementation of kmeans ++ from kmeans. The main difference is in the
    initialization of the cluster centers
    """

    def __init__(self, n_clusters, embed_dim,  n_labels=526):
        super().__init__(n_clusters, embed_dim, n_labels=n_labels)

        self.number_initialized = tf.Variable(tf.zeros(1, dtype=tf.int32))
        self.centroids = tf.Variable(tf.zeros((n_clusters, embed_dim)), trainable=False)

    @property
    def initialized(self):
        return tf.equal(tf.reduce_sum(self.number_initialized), self.n_clusters)

    def initialize(self, points, eps=10**-8):
        """

        :param points:
        :return:
        """
        centroids = tf.identity(self.centroids)

        number_initialized = tf.cast(tf.reduce_sum(self.number_initialized), tf.int32)
        dist = self.compute_similarity(points, centroids[:number_initialized + 1, ...])
        dist = tf.reduce_min(dist, axis=1)
        dist_norm = dist / (tf.reduce_sum(dist) + eps)

        i = tf.cond(tf.equal(number_initialized, 0), lambda: 0, lambda: tf.cast(tf.argmax(dist_norm), dtype=tf.int32))

        cen = tf.tensor_scatter_nd_update(centroids, tf.expand_dims(self.number_initialized, 1), tf.expand_dims(tf.gather(points, i), 0))
        self.centroids.assign(cen)
        self.number_initialized.assign_add([1])


class MahaKmeans(KMeansPlusPlus):

    def __init__(self, n_clusters, embed_dim,  n_labels=526):
        super().__init__(n_clusters, embed_dim, n_labels=n_labels)

        self.centroids_covariance = tf.Variable(tf.eye(embed_dim, num_columns=embed_dim, batch_shape=[n_clusters]),
                                                        trainable=False)

        #initialize covariance with identity matrix:
        self.running_centroids_covariance = tf.Variable(tf.zeros((n_clusters, embed_dim, embed_dim)),
                                                trainable=False)

    def compute_similarity(self, x, c, covariance=None):
        """
        compute euclidean distance between x and c along the second dimension
        :param x: (B, dim)
        :param c: (ncluster, dim)
        :return: (B, nclusters)
        """
        x_exp = tf.expand_dims(x, 1)
        c_exp = tf.expand_dims(c, 0)

        if covariance is None:
            dist = tf.reduce_sum(tf.math.square(x_exp - c_exp), -1)
        else:
            cov = tf.expand_dims(covariance, 0) #(1, ncluster, dim, dim)
            dist = tf.linalg.matvec(tf.linalg.inv(cov), x_exp - c_exp) # (B, nlcuster, dim)

            dist = tf.reduce_sum((x_exp - c_exp) * dist, 2)

        return dist

    def assign(self, points):
        """

        :param training:
        :param labels:
        :param eps:
        :param points:
        :return:
        """
        centroids = self.centroids
        covariance = self.centroids_covariance

        distance = self.compute_similarity(points, centroids, covariance=covariance)
        centroid_assignment = tf.argmin(distance, 1)

        return centroid_assignment

    def update(self, points, centroid_assignment, labels, eps=10 ** (-8)):
        """
        :param eps:
        :param labels:
        :param centroid_assignment:
        :param points:
        :return:
        """
        centroids = self.centroids
        k = centroids.shape[0]

        new_sum = tf.math.unsorted_segment_sum(points, centroid_assignment, k)
        new_n = tf.math.unsorted_segment_sum(tf.ones(tf.shape(points)[0]), centroid_assignment, k)

        new_labels = tf.math.unsorted_segment_sum(labels, centroid_assignment, k)
        new_labels = tf.cast(new_labels, dtype=tf.float32)

        deviations = points - tf.nn.embedding_lookup(centroids, centroid_assignment)
        deviations = tf.expand_dims(deviations, 2)

        cov = tf.matmul(deviations, tf.transpose(deviations, perm=[0, 2, 1]))
        new_covariance = tf.math.unsorted_segment_sum(cov, centroid_assignment, k)
        new_covariance += self.running_centroids_covariance

        new_sum = self.running_centroids + new_sum
        new_n = self.running_N + new_n

        new_labels = self.running_labels + new_labels
        self.running_labels.assign(new_labels)

        self.running_centroids.assign(new_sum)
        self.running_N.assign(new_n)
        self.running_centroids_covariance.assign(new_covariance)

    def reset_centroids(self, eps=10**-8):

        """
        Reset running counts after assigning them to state variables
        :return: dist
        """
        n = tf.expand_dims(self.running_N, 1) + eps
        self.centroids.assign(self.running_centroids / n)
        id = tf.eye(self.embed_dim, num_columns=self.embed_dim, batch_shape=[self.n_clusters])

        self.centroids_covariance.assign(self.running_centroids_covariance / tf.expand_dims(n, 2) + id)
        self.cluster_labels.assign(self.running_labels / n)
        self.N.assign(self.running_N)
        self.center.assign(self.running_center)
        self.dispersion.assign(self.running_dispersion)
        self.dispersion_labels.assign(self.running_dispersion_labels)

        self.running_N.assign(tf.zeros(self.n_clusters))
        self.running_centroids.assign(tf.zeros((self.n_clusters, self.embed_dim)))
        self.running_labels.assign(tf.zeros((self.n_clusters, self.n_labels)))
        self.running_center.assign(tf.Variable(tf.zeros(self.embed_dim), trainable=False))
        self.running_dispersion.assign(tf.Variable(tf.zeros(self.n_clusters), trainable=False))
        self.running_dispersion_labels.assign(tf.Variable(tf.zeros(self.n_clusters), trainable=False))
        self.running_centroids_covariance.assign(tf.Variable(tf.zeros((self.n_clusters, self.embed_dim, self.embed_dim)),
                                                        trainable=False))