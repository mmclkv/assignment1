import autodiff as ad
import numpy as np
import random
import matplotlib.pyplot as plt


def generate_dataset(w1, w2, b, x_limit=[0.0, 5.0], y_limit=[0.0, 5.0], point_num=100):
    """ Generate data points that lie on the two side of w1*x1 + w2*x2 = b """
    x = np.zeros((point_num, 2))
    y = np.zeros((point_num, 1))

    for i in range(point_num):
        x[i, 0] = random.uniform(*x_limit)
        x[i, 1] = random.uniform(*y_limit)
        if w1 * x[i, 0] + w2 * x[i, 1] + b > 0:
            y[i] = 1

    return x, y


def draw(W_val, x_t, y_t):
    """ Plot the decision superplane """
    w1, w2, b = W_val
    plt.plot([0, -b / w1], [-b / w2, 0], color='black')
    plt.scatter(x_t[:, 0], x_t[:, 1], y_t, color='red')
    plt.scatter(x_t[:, 0], x_t[:, 1], 1 - y_t, color='blue')
    plt.show()


def main():
    # Generate dataset and initial weight
    x_t, y_t = generate_dataset(1, 1, -5, point_num=50)

    # add extra dim to build homogenous coordinates
    x_t = np.concatenate((x_t, np.ones((x_t.shape[0], 1))), axis=1)
    W_val = np.random.rand(3, 1)

    # draw initial decision superplane
    draw(W_val, x_t, y_t)

    # Create the model
    x = ad.Variable(name = 'x')
    W = ad.Variable(name = 'W')
    y = ad.sigmoid_op(ad.matmul_op(x, W))

    # Define loss
    y_ = ad.Variable(name = 'y_')
    cross_entropy = ad.reduce_mean_op(
        -ad.reduce_sum_op(
            y_ * ad.log_op(y) + (1 - y_) * ad.log_op(1 - y),
            reduction_indices=[1]
        )
    )

    # Update rule
    learning_rate = 0.05
    W_grad, = ad.gradients(cross_entropy, [W])
    W_train_step = W - learning_rate * W_grad

    # Training
    executor = ad.Executor([cross_entropy, y, W_train_step])
    steps = 200
    for i in range(steps):
        loss_val, y_val, W_val = executor.run(
            feed_dict = {
                x: x_t,
                y_: y_t,
                W: W_val,
            }
        )
        print("Step {}: loss: {}".format(i + 1, loss_val))

    # draw trained decision superplane
    draw(W_val, x_t, y_t)


if __name__ == "__main__":
    main()
