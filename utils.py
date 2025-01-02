import numpy as np
import matplotlib.pyplot as plt


def vizualize_results(iters, cost, i2_est ):
    fig, ax = plt.subplots()
    ax.plot(np.arange(iters), cost)
    ax.set(xlabel='Iters', ylabel='Cost', title='Cost along iters')

    # fig, ax1 = plt.subplots()
    # ax1.plot(np.arange(iters), phase_err)
    # ax1.set(xlabel='Iters', ylabel='Cost', title='Phase error along iters')


    fig1, axs = plt.subplots(figsize=(15, 8))

    # axs 是 2x2 的数组，通过索引访问每个子图
    im1 = axs.imshow(i2_est)
    fig1.colorbar(im1, ax=axs)
    axs.set_title("The predicted phase at z_obj")

    # im2 = axs[1].imshow(i2)
    # fig1.colorbar(im2, ax=axs[1])
    # axs[1].set_title("The actual phase at z_obj")
    #
    # im3 = axs[2].imshow(i0)
    # axs[2].set_title("The phase error")
    # fig1.colorbar(im3, ax=axs[2])



    plt.tight_layout()  # 调整布局避免重叠
    plt.show()
