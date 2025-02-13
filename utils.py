import numpy as np
import matplotlib.pyplot as plt


def vizualize_results(iters, cost, i2_est, i2, i0, ph0_est):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost)
    ax.set(xlabel='Iters', ylabel='Cost', title='Cost along iters')

    fig1, axs = plt.subplots(2, 2, figsize=(15, 8))

    # axs 是 2x2 的数组，通过索引访问每个子图
    im1 = axs[0, 0].imshow(i2_est)
    fig1.colorbar(im1, ax=axs[0, 0])
    axs[0, 0].set_title("Predicted intensity at z2")

    im2 = axs[0, 1].imshow(i2)
    fig1.colorbar(im2, ax=axs[0, 1])
    axs[0, 1].set_title("The actual intensity at z2")

    im3 = axs[1, 0].imshow(i0)
    axs[1, 0].set_title("Predicted intensity at z0")
    fig1.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(ph0_est)
    axs[1, 1].set_title("Predicted phase at z0")
    fig1.colorbar(im4, ax=axs[1, 1])

    plt.tight_layout()  # 调整布局避免重叠
    plt.show()

def vizualize_results_1(iters, cost, i2_est):

    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost)
    ax.set(xlabel='Iters', ylabel='Cost', title='Cost along iters')

    fig1, axs = plt.subplots(figsize=(15, 8))

    # axs 是 2x2 的数组，通过索引访问每个子图
    im1 = axs.imshow(i2_est)
    fig1.colorbar(im1, ax=axs)
    axs.set_title("The predicted phase at z_obj")

    plt.tight_layout()  # 调整布局避免重叠
    plt.show()


def vizualize_results_2(iters, cost, i2_est):
    # 绘制 Cost 随迭代变化的图
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost)
    ax.set(xlabel='Iters', ylabel='Cost', title='Cost along iters')

    # 提取中心区域（ROI）
    center_size_x, center_size_y = 100, 100  # 中心区域的大小
    center_x, center_y = i2_est.shape[0] // 2+50, i2_est.shape[1] // 2-25  # 图像的中心
    roi = i2_est[
          center_x - center_size_x // 2:center_x + center_size_x // 2,
          center_y - center_size_y // 2:center_y + center_size_y // 2
          ]

    # 显示中心区域的图像
    fig1, axs = plt.subplots(figsize=(15, 8))
    im1 = axs.imshow(roi, cmap='jet', vmin=np.min(roi), vmax=np.max(roi), interpolation='nearest', origin='lower')
    fig1.colorbar(im1, ax=axs)
    axs.set_title("The predicted phase at z_obj (Center Region)")

    plt.tight_layout()  # 调整布局避免重叠
    plt.show()

