# import numpy as np
# # d_pool = np.arange(1, 9, 1)
# # # print((d_pool))
# lin=
# # # for i in enumerate(d_pool):
# # # 	print((i))
# # # print(lin[('alpha')])
# if hasattr(lin, '2341'):
# 	print(12)
# output = '%s：%d阶，系数为：' % (1, 6)
# idx = output.find('系数')
# print(output[:idx])
# output = output[:idx] + ('alpha') + output[idx:]
# print(output)
# a=range(5)
# print(a)
from sklearn import svm
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



# 生成一个有两个特征、包含两种类别的数据集
def plot_hyperplane(clf, X, y, 
                    h=0.02, 
                    draw_sv=True, 
                    title='hyperplan'):
    # create a mesh to plot in
    # fig = plt.figure()
    # ax = Axes3D(fig)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) # SVM的分割超平面
    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    print(Z.shape)
    np.savetxt('12.txt', Z, fmt='%d', delimiter='')
    np.savetxt('13.txt', xx, fmt='%d', delimiter='')
    np.savetxt('14.txt', yy, fmt='%d', delimiter='')
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)


    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    print(labels)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    # 画出支持向量
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')
    plt.show()
    # fig = plt.figure()
    # ax = Axes3D(fig)

    # ax.plot_surface(xx, yy, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
    # ax.set_zlim(-6, 6)

    # plt.show()
    # ax.plot_surface(X, Y, (Z), rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))





X, y = make_blobs(n_samples=20, centers=2, 
                  random_state=0, cluster_std=0.8)
clf = svm.SVC(C=1.0, kernel='linear')
print(X,y)
clf.fit(X, y)
# print(clf)
# plt.figure(figsize=(12, 4), dpi=144)
plot_hyperplane(clf, X, y, h=0.01, 
                title='Maximum Margin Hyperplan')

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# x = np.array([1,2,3])
# y = np.array([2,3,4])
# xx,yy=np.meshgrid(x,y)
# print(xx,yy)
# z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# # z=np.sin(xx*yy)#设置z由xx+yy生成，当然也可以是其他函数如sin（xx+yy）等
# z=z.reshape(xx.shape)#z数组的大小必须和xx,yy大小一致
# plt.contourf(xx, yy, z, cmap='hot', alpha=0.5)#3代表levels=3,绘制3条等高线，等高线的间距由方法自动生成
# plt.scatter(xx.reshape(9),yy.reshape(9),c='y', marker='x')

# plt.show()
# fig = plt.figure()
# ax = Axes3D(fig)

# ax.plot_surface(xx, yy, z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# plt.show()
# ax.set_zlim(-6, 6)
# print((xx.reshape(9)).shape)


