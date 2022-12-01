"""
This file includes the plots that we can use over the application
"""


import matplotlib.pyplot as plt

dlc = dict(dlblue='#0096ff',
           dlorange='#FF9300',
           dldarkred='#C00000',
           dlmagenta='#FF40FF',
           dlpurple='#7030A0',
           dldarkblue='#0D5BDC')


def plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree):
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax.set_title("Poor Performance on Test Data",fontsize = 12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(X_train, y_train, color="red", label="train")
    ax.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.plot(x, y_pred,  lw=0.5, label=f"predicted, degree={degree}")
    ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plt_optimal_degree(X_train, y_train, X_cv, y_cv, x, y_pred, x_ideal, y_ideal, err_train, err_cv, optimal_degree, max_degree):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax[0].set_title("predictions vs data",fontsize = 12)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[0].plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax[0].scatter(X_train, y_train, color="red",           label="train")
    ax[0].scatter(X_cv, y_cv,       color=dlc["dlorange"], label="cv")
    ax[0].set_xlim(ax[0].get_xlim())
    ax[0].set_ylim(ax[0].get_ylim())
    for i in range(0,max_degree):
        ax[0].plot(x, y_pred[:, i],  lw=0.5, label=f"{i+1}")
    ax[0].legend(loc='upper left')

    ax[1].set_title("error vs degree", fontsize=12)
    cpts = list(range(1, max_degree+1))
    ax[1].plot(cpts, err_train[0:], marker='o', label="train error", lw=2,  color=dlc["dlblue"])
    ax[1].plot(cpts, err_cv[0:],    marker='o', label="cv error",  lw=2, color=dlc["dlorange"])
    ax[1].set_ylim(*ax[1].get_ylim())
    ax[1].axvline(optimal_degree, lw=1, color=dlc["dlmagenta"])
    ax[1].annotate("optimal degree", xy=(optimal_degree, 80000), xycoords='data',
                xytext=(0.3, 0.8), textcoords='axes fraction', fontsize=10,
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3",
                                   color=dlc['dldarkred'], lw=1))
    ax[1].set_xlabel("degree")
    ax[1].set_ylabel("error")
    ax[1].legend()
    fig.suptitle("Find Optimal Degree", fontsize=12)
    plt.tight_layout()

    plt.show()


def plt_tune_regularization(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, optimal_reg_idx, lambda_range):
    fig, ax = plt.subplots(1,2,figsize=(8,4))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax[0].set_title("predictions vs data", fontsize = 12)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[0].scatter(X_train, y_train, color="red",           label="train")
    ax[0].scatter(X_cv, y_cv,       color=dlc["dlorange"], label="cv")
    ax[0].set_xlim(ax[0].get_xlim())
    ax[0].set_ylim(ax[0].get_ylim())
#   ax[0].plot(x, y_pred[:,:],  lw=0.5, label=[f"$\lambda =${i}" for i in lambda_range])
    for i in (0, 3, 7, 9):
        ax[0].plot(x, y_pred[:, i],  lw=0.5, label=f"$\lambda =${lambda_range[i]}")
    ax[0].legend()

    ax[1].set_title("error vs regularization", fontsize=12)
    ax[1].plot(lambda_range, err_train[:], label="train error", color=dlc["dlblue"])
    ax[1].plot(lambda_range, err_cv[:],    label="cv error",    color=dlc["dlorange"])
    ax[1].set_xscale('log')
    ax[1].set_ylim(*ax[1].get_ylim())
    opt_x = lambda_range[optimal_reg_idx]
    ax[1].vlines(opt_x, *ax[1].get_ylim(), color = "black", lw=1)
    ax[1].annotate("optimal lambda", (opt_x,150000), xytext=(-80, 10), textcoords="offset points",
                  arrowprops={'arrowstyle': 'simple'})
    ax[1].set_xlabel("regularization (lambda)")
    ax[1].set_ylabel("error")
    fig.suptitle("Tuning Regularization", fontsize = 12)
    ax[1].text(0.05, 0.44, "High\nVariance", fontsize=12, ha='left', transform=ax[1].transAxes, color=dlc["dlblue"])
    ax[1].text(0.95, 0.44, "High\nBias",    fontsize=12, ha='right', transform=ax[1].transAxes, color=dlc["dlblue"])
    ax[1].legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plt_tune_m(X_train, y_train, X_cv, y_cv, x, y_pred, err_train, err_cv, m_range, degree):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax[0].set_title("predictions vs data", fontsize=12)
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")

    ax[0].scatter(X_train, y_train, color="red", s=3, label="train", alpha=0.4)
    ax[0].scatter(X_cv, y_cv, color=dlc["dlorange"], s=3, label="cv", alpha=0.4)
    ax[0].set_xlim(ax[0].get_xlim())
    ax[0].set_ylim(ax[0].get_ylim())
    for i in range(0, len(m_range), 3):
        ax[0].plot(x, y_pred[:, i], lw=1, label=f"$m =${m_range[i]}")
    ax[0].legend(loc='upper left')
    ax[0].text(0.05, 0.5, f"degree = {degree}", fontsize=10, ha='left', transform=ax[0].transAxes, color=dlc["dlblue"])

    ax[1].set_title("error vs number of examples", fontsize=12)
    ax[1].plot(m_range, err_train[:], label="train error", color=dlc["dlblue"])
    ax[1].plot(m_range, err_cv[:], label="cv error", color=dlc["dlorange"])
    ax[1].set_xlabel("Number of Examples (m)")
    ax[1].set_ylabel("error")
    fig.suptitle("Tuning number of examples", fontsize=12)
    ax[1].text(0.05, 0.5, "High\nVariance", fontsize=12, ha='left', transform=ax[1].transAxes, color=dlc["dlblue"])
    ax[1].text(0.95, 0.5, "Good \nGeneralization", fontsize=12, ha='right', transform=ax[1].transAxes,
               color=dlc["dlblue"])
    ax[1].legend()
    plt.tight_layout()
    plt.show()
