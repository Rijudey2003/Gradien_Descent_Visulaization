import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def err_plot(l, u, n, x, y, act, lossf, dfs=None, pause=0.0001):
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    w_vals = np.linspace(l, u, n)
    b_vals = np.linspace(l, u, n)
    W, B = np.meshgrid(w_vals, b_vals)
    err = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_vec = np.array([[B[i, j]], [W[i, j]]])
            z = np.dot(x, w_vec)

            if act == 'sigmoid':
                z = np.clip(z, -500, 500)
                pred = 1.0 / (1.0 + np.exp(-z))
                if lossf == 'squared':
                    err[i, j] = np.mean((pred - y) ** 2)
                elif lossf == 'cross_entropy':
                    err[i, j] = -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
            elif act == 'linear':
                pred = z
                if lossf == 'squared':
                    err[i, j] = np.mean((pred - y) ** 2)

 
    fig = plt.figure(figsize=(14, 6))

    # 2D contour subplot
    ax2d = fig.add_subplot(1, 2, 1)
    contour = ax2d.contourf(W, B, err, levels=30, cmap='coolwarm')
    fig.colorbar(contour, ax=ax2d, label='Loss')
    ax2d.set_xlabel('Weight')
    ax2d.set_ylabel('Bias')
    ax2d.set_title('2D Contour of Loss Surface')

    # 3D surface subplot
    ax3d = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax3d.plot_surface(W, B, err, cmap='coolwarm', edgecolor='none', alpha=0.8)
    fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=10, label='Loss')
    ax3d.set_xlabel('Weight')
    ax3d.set_ylabel('Bias')
    ax3d.set_zlabel('Loss')
    ax3d.set_title('3D Surface of Loss Surface')

    
    if dfs is not None:
        plt.ion()
        for i in range(len(dfs[0]['df'])):
            for opt in dfs:
                df = opt['df']
                color = opt.get('color', 'black')
                label = opt.get('label', '')

                if i < len(df):
                    w_val = df['weight'].iloc[i]
                    b_val = df['bias'].iloc[i]
                    loss_val = df['loss'].iloc[i]
                    epoch = df['epoch'].iloc[i]
                    iteration = df['iteration'].iloc[i]

                    # 2D point
                    ax2d.scatter(w_val, b_val, color=color, s=10, label=label if i == 0 else None)

                    # 3D point
                    ax3d.scatter(w_val, b_val, loss_val, color=color, s=10)

            fig.suptitle(f"Epoch: {epoch} | Iteration: {iteration}", fontsize=14)
            plt.pause(pause)

        # Show legend only once
        handles, labels = ax2d.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2d.legend(by_label.values(), by_label.keys())

        plt.ioff()
        plt.show()
    else:
        plt.show()

    return {'2d': (fig, ax2d), '3d': (fig, ax3d)}

  
def grad_des(w, b, X, Y, eta, max_epoch, act, lossf, learn_opt='vanilla', update='vanilla', batch_size=None):
    loss = []
    epo = []
    iteration = []
    weight = []
    bias = []
    
    grad = 0.0
    v = 0.0
    m = 0.0

    X = np.array(X)
    Y = np.array(Y)
    X_design = np.hstack((np.ones((X.shape[0], 1)), X))
    w = np.vstack((b, np.array(w).reshape(-1, 1))).astype(float)
    up = np.zeros_like(w)
    if batch_size is None:
        batch_size = X.shape[0]

    # Store initial values before training
    bias.append(w[0][0])
    weight.append(w[1][0])
    epo.append(0)
    iteration.append(0)
    loss.append(np.nan)  # No loss yet

    for epoch in range(max_epoch):      
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        itr = 0
        for j in range(0, X.shape[0], batch_size):
            itr += 1
            batch_idx = indices[j:j + batch_size]
            xb = X_design[batch_idx]
            yb = Y[batch_idx]
            if update == 'nag':
                w_lookahead = w - 0.9 * up
                z = np.dot(xb, w_lookahead)
            else:
                z = np.dot(xb, w)

            if act == 'sigmoid':
                z = np.clip(z, -500, 500)
                o = 1.0 / (1.0 + np.exp(-z))
                if lossf == 'squared':
                    loss_val = np.mean((o - yb) ** 2)
                    grad = np.dot(xb.T, (o - yb) * o * (1 - o)) / xb.shape[0]
                elif lossf == 'cross_entropy':
                    loss_val = -np.mean(yb * np.log(o + 1e-8) + (1 - yb) * np.log(1 - o + 1e-8))
                    grad = np.dot(xb.T, o - yb) / xb.shape[0]
            elif act == 'linear':
                o = z
                if lossf == 'squared':
                    loss_val = np.mean((o - yb) ** 2)
                    grad = np.dot(xb.T, (o - yb)) / xb.shape[0]
                else:
                    print('Only squared loss is supported for linear activation.')
                    return None

            loss.append(loss_val)
            bias.append(w[0][0])
            weight.append(w[1][0])
            epo.append(epoch)
            iteration.append(itr)

            if learn_opt == 'vanilla':
                rate = eta
            elif learn_opt == 'adagrad':
                v += grad ** 2
                rate = eta / np.sqrt(v + 1e-8)
            elif learn_opt == 'rmsprop':
                v = 0.95 * v + (1 - 0.95) * grad ** 2
                rate = eta / np.sqrt(v + 1e-8)
            elif learn_opt == 'adam':
                m = 0.9 * m + (1 - 0.9) * grad
                v = 0.999 * v + (1 - 0.999) * grad ** 2
                mcap = m / (1 - 0.9)
                vcap = v / (1 - 0.999)
                rate = (eta / np.sqrt(vcap + 1e-8)) * mcap / grad

            if update == 'vanilla':
                w = w - eta * grad
            elif update == 'momentum':
                up = 0.9 * up + eta * grad
                w = w - up
            elif update == 'nag':
                up = 0.9 * up + eta * w_lookahead
                w = w - up

    df = pd.DataFrame({'epoch': epo, 'iteration': iteration, 'loss': loss, 'weight': weight, 'bias': bias})
    return {'weights': w, 'loss_df': df}
  

X = np.random.randn(100, 1)
Y = 1 / (1 + np.exp(-np.dot(X, np.array([[2]]))))
eta = 0.1
max_epoch =       100
act =            'sigmoid'
lossf =          'cross_entropy'
#learn_opt  =    'adam'
#update =        'momentum'
batch_size =    50
w = -12
b = -12

df_adam = grad_des(-12, -10, X, Y, eta, max_epoch, act, lossf, learn_opt='vanilla', update='momentum', batch_size=batch_size)['loss_df']
df_vanilla = grad_des(-5, -3, X, Y, eta, max_epoch, act, lossf, learn_opt='vanilla', update='vanilla', batch_size=batch_size)['loss_df']

l = min(df_adam['weight'].min(), df_vanilla['weight'].min(), df_adam['bias'].min(), df_vanilla['bias'].min()) - 5
u = max(df_adam['weight'].max(), df_vanilla['weight'].max(), df_adam['bias'].max(), df_vanilla['bias'].max()) + 5

err_plot(
    l, u, 100, X, Y, act, lossf,
    dfs=[
        {'df': df_adam, 'label': 'Adam', 'color': 'red'},
        {'df': df_vanilla, 'label': 'Vanilla', 'color': 'blue'}
    ]
)
