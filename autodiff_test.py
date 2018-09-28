import autodiff as ad
import numpy as np

def test_identity():
    x2 = ad.Variable(name = "x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))

def test_add_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 + x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))

def test_sub():
    x2 = ad.Variable(name = "x2")
    y1 = x2 - 5
    y2 = 5 - x2

    grad_y1_x2, = ad.gradients(y1, [x2])
    grad_y2_x2, = ad.gradients(y2, [x2])

    executor = ad.Executor([y1, y2, grad_y1_x2, grad_y2_x2])
    x2_val = 2 * np.ones(3)
    y1_val, y2_val, grad_y1_x2_val, grad_y2_x2_val = executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y1, ad.Node)
    assert isinstance(y2, ad.Node)
    assert np.array_equal(y1_val, x2_val - 5)
    assert np.array_equal(y2_val, 5 - x2_val)
    assert np.array_equal(grad_y1_x2_val, np.ones_like(x2_val))
    assert np.array_equal(grad_y2_x2_val, (-1) * np.ones_like(x2_val))

def test_mul_by_const():
    x2 = ad.Variable(name = "x2")
    y = 5 * x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    y_val, grad_x2_val= executor.run(feed_dict = {x2 : x2_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * 5)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val) * 5)

def test_add_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val + x3_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))
    assert np.array_equal(grad_x3_val, np.ones_like(x3_val))

def test_mul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)

def test_add_mul_mix_1():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x1 + x2 * x3 * x1

    grad_x1, grad_x2, grad_x3 = ad.gradients(y, [x1, x2, x3])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val * x1_val)
    assert np.array_equal(grad_x3_val, x2_val * x1_val)

def test_add_mul_mix_2():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    x4 = ad.Variable(name = "x4")
    y = x1 + x2 * x3 * x4

    grad_x1, grad_x2, grad_x3, grad_x4 = ad.gradients(y, [x1, x2, x3, x4])

    executor = ad.Executor([y, grad_x1, grad_x2, grad_x3, grad_x4])
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    y_val, grad_x1_val, grad_x2_val, grad_x3_val, grad_x4_val = executor.run(feed_dict = {x1 : x1_val, x2: x2_val, x3 : x3_val, x4 : x4_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x1_val + x2_val * x3_val * x4_val)
    assert np.array_equal(grad_x1_val, np.ones_like(x1_val))
    assert np.array_equal(grad_x2_val, x3_val * x4_val)
    assert np.array_equal(grad_x3_val, x2_val * x4_val)
    assert np.array_equal(grad_x4_val, x2_val * x3_val)

def test_add_mul_mix_3():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    z_val = x2_val * x2_val + x2_val + x3_val + 3
    expected_yval = z_val * z_val + x3_val
    expected_grad_x2_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1)
    expected_grad_x3_val = 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1
    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)

def test_grad_of_grad():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = x2 * x2 + x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])
    grad_x2_x2, grad_x2_x3 = ad.gradients(grad_x2, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3, grad_x2_x2, grad_x2_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val, grad_x2_x2_val, grad_x2_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = x2_val * x2_val + x2_val * x3_val
    expected_grad_x2_val = 2 * x2_val + x3_val
    expected_grad_x3_val = x2_val
    expected_grad_x2_x2_val = 2 * np.ones_like(x2_val)
    expected_grad_x2_x3_val = 1 * np.ones_like(x2_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)
    assert np.array_equal(grad_x2_x2_val, expected_grad_x2_x2_val)
    assert np.array_equal(grad_x2_x3_val, expected_grad_x2_x3_val)

def test_matmul_two_vars():
    x2 = ad.Variable(name = "x2")
    x3 = ad.Variable(name = "x3")
    y = ad.matmul_op(x2, x3)

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]]) # 2x3

    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict = {x2: x2_val, x3: x3_val})

    expected_yval = np.matmul(x2_val, x3_val)
    expected_grad_x2_val = np.matmul(np.ones_like(expected_yval), np.transpose(x3_val))
    expected_grad_x3_val = np.matmul(np.transpose(x2_val), np.ones_like(expected_yval))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x2_val, expected_grad_x2_val)
    assert np.array_equal(grad_x3_val, expected_grad_x3_val)

def test_div_two_vars():
    x1 = ad.Variable(name = "x1")
    x2 = ad.Variable(name = "x2")
    y = ad.div_op(x1, x2)

    grad_x1, grad_x2 = ad.gradients(y, [x1, x2])
    executor = ad.Executor([y, grad_x1, grad_x2])
    x1_val = np.array([1.0, 2.0])
    x2_val = np.array([3.0, 4.0])

    y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict = {x1: x1_val, x2: x2_val})
    expected_yval = np.array([0.33333333, 0.5])
    expected_grad_x1_val = np.array([0.33333333, 0.25])
    expected_grad_x2_val = np.array([-0.11111111, -0.125])

    assert isinstance(y, ad.Node)
    assert np.allclose(y_val, expected_yval)
    assert np.allclose(grad_x1_val, expected_grad_x1_val)
    assert np.allclose(grad_x2_val, expected_grad_x2_val)

def test_log():
    x1 = ad.Variable(name = "x1")
    y = ad.log_op(x1)

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = np.array([[1.0, 2.0], [3.0, 4.0]])

    y_val, grad_x1_val = executor.run(feed_dict = {x1: x1_val})

    expected_yval = np.log(x1_val)
    expected_grad_x1_val = np.reciprocal(x1_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x1_val, expected_grad_x1_val)

def test_exp():
    x1 = ad.Variable(name = "x1")
    y = ad.exp_op(x1)

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = np.array([[1.0, 2.0], [3.0, 4.0]])

    y_val, grad_x1_val = executor.run(feed_dict = {x1: x1_val})

    expected_yval = np.exp(x1_val)
    expected_grad_x1_val = np.exp(x1_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x1_val, expected_grad_x1_val)

def test_sigmoid():
    x1 = ad.Variable(name = "x1")
    y = ad.sigmoid_op(x1)

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = np.array([[1.0, 2.0], [3.0, 4.0]])

    y_val, grad_x1_val = executor.run(feed_dict = {x1: x1_val})

    expected_yval = 1 / (1 + np.exp(-x1_val))
    expected_grad_x1_val = np.exp(-x1_val) / ((1 + np.exp(-x1_val)) ** 2)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x1_val, expected_grad_x1_val)

def test_reducesum():
    x1 = ad.Variable(name = "x1")
    y = ad.reduce_sum_op(x1, reduction_indices=[1])

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3],[1, 2, 3],[1, 2, 3]])

    y_val, grad_x1_val = executor.run(feed_dict = {x1: x1_val})

    expected_yval = np.array([6, 6, 6, 6, 6])
    expected_grad_x1_val = np.ones((5, 3))

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.array_equal(grad_x1_val, expected_grad_x1_val)

def test_reducemean():
    x1 = ad.Variable(name = "x1")
    y = ad.reduce_mean_op(x1, reduction_indices=[1])

    grad_x1, = ad.gradients(y, [x1])

    executor = ad.Executor([y, grad_x1])
    x1_val = np.array([[1, 2, 3],[1, 2, 3],[1, 2, 3],[1, 2, 3],[1, 2, 3]])

    y_val, grad_x1_val = executor.run(feed_dict = {x1: x1_val})

    expected_yval = np.array([2, 2, 2, 2, 2])
    expected_grad_x1_val = np.ones((5, 3)) / 3.0

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, expected_yval)
    assert np.allclose(grad_x1_val, expected_grad_x1_val)
