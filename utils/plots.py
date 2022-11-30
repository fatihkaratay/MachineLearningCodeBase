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
