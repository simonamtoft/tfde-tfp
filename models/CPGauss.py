import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class CPGaussian(tf.keras.Model):
    """A CP Decomposition (Diagonal Gaussian Mixture Model) consisting of independent univariate Gaussians

    Parameters
    ----------
    K : int > 0
    Amount of components
    M : int > 0
    Amount of dimensions pr. component
    """
    def __init__(self, K, M):
        super(CPGaussian, self).__init__()
        self.K = K
        self.M = M
        self.w_logits = tf.Variable([1.]*K, name="logits")
        self.locs = [
            [
                tf.Variable(
                    tf.random.uniform([],-4, 4), 
                    name="mu_{},{}".format(i,j)
                ) for j in range(self.M)
            ] for i in range(self.K)
        ]
        self.scales = [
            [
                tf.Variable(
                    0.5,
                    name="sigma_{},{}".format(i,j)
                ) for j in range(self.M)
            ] for i in range(self.K)
        ]
        self.distributions = [
            [
                tfd.Normal(self.locs[i][j], self.scales[i][j]) for j in range(M)
            ] for i in range(K)
        ]

        self.components = [
            tfd.Blockwise(dists) 
            for dists in self.distributions
        ]
        self.density = tfd.Mixture(
            cat=tfd.Categorical(logits=self.w_logits),
            components=self.components
        )
        return None

    def call(self, x):
        log_likelihoods = self.density.log_prob(x)
        return log_likelihoods

    def initialize(self, centroids, scales):
        for i in range(self.K):
            for j in range(self.M):
                self.locs[i][j].assign(centroids[i][j])
                self.scales[i][j].assign(scales[i][j])

    # @tf.function
    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            log_likelihoods = self(data)
            loss = -tf.reduce_mean(log_likelihoods)
        tvars = self.trainable_variables
        gradients = tape.gradient(loss, tvars)
        optimizer.apply_gradients(zip(gradients, tvars))
        return loss

