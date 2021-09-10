from TrianglePlot import *

np.random.seed(2342342)
Ndim = 3
M = 15000

# generate 3 samples (for this demo wihout any underlying meaning)
ranges = np.array([[-10.0, 10.0] for i in range(Ndim)])
plot_ranges = np.array([[-10.0, 10.0] for i in range(Ndim)])

# generate sample 1
y12_1 = np.random.multivariate_normal(np.array([-5.0, -5.0]),
                                      np.array([[4.0, -2.95],
                                                [-2.95, 4.0]]), M / 6)

y12_2 = np.random.multivariate_normal(np.array([5.0, 5.0]),
                                      np.array([[4.0, -0.05],
                                                [-0.05, 4.0]]), M - M / 6)

y13 = np.random.uniform(ranges[2, 0], ranges[2, 1], M)[:, np.newaxis]
y12 = np.append(y12_1, y12_2, axis=0)
y1 = np.append(y12, y13, axis=1)
id1 = True

# reduce the above sample to what is within the considered range
for i in range(Ndim):
    id1 = id1 * (y1[:, i] >= ranges[i, 0]) * (y1[:, i] <= ranges[i, 1])
y1 = y1[id1]
w1 = np.ones((M))[id1]

# generate sample 2
mean = np.array([-5.0, 5.0, 5.0])
cov = np.array([[1.0, -1.95, 0], [-1.95, 16.0, 0.0], [0.0, 0.0, 1.0]])
y2 = np.random.multivariate_normal(mean, cov, M)
id2 = True
for i in range(Ndim):
    id2 = id2 * (y2[:, i] >= ranges[i, 0]) * (y2[:, i] <= ranges[i, 1])
y2 = y2[id2]
w2 = np.ones((M))[id2]

# generate sample 3 weights
y3 = np.random.uniform(ranges[2, 0], ranges[2, 1], (M, 3))
w3 = np.array([np.exp(-0.5 * y3[u].dot(np.linalg.inv(cov).dot(y3[u])))
               for u in range(M)])

# create list of the samples
s = [[y1, w1], [y2, w2], [y3, w3]]

Nplot = 50
dims_to_plot = [0, 1, 2]  # [i for i in range(Ndim)]
labels = [r"$x_%d$" % i for i in range(Ndim)]
slabels = [r"sample $%d$" % i for i in range(3)]

# s is a list of samples and weights.
# dims_to_plot is a list of the components that are shown.
# ranges are the bounds of the samples.
# N is the number of bins for plotting and determining the contours.
# refl is 0 or 1 and turns off and on the reflection of samples at the
# boundaries (useful for finite ranges, but computationally slower).
# plot ranges must always be contained within orginal ranges of the parameters.
triangl_plot(s, dims_to_plot, ranges, Nplot, plot_ranges,
             labels=labels, refl=0, slabels=slabels)

plt.show()
