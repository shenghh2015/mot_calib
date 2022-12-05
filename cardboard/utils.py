import numpy as np
import os
import json

def vec_cos(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim

def compute_angles(a, b):
    cos_sim   = vec_cos(a, b)
    rad_angle = np.arccos(cos_sim)
    angle     = rad_to_deg(rad_angle)
    return angle

def rad_to_deg(rad_vec):
    return rad_vec/np.pi * 180

def deg_to_rad(deg_vec):
    return deg_vec/180 * np.pi

def gen_dir(folder):
    if not os.path.exists(folder):
        os.system('mkdir -p {}'.format(folder))

def get_box_pts(boxes, 
                loc = 'bottom'):
    if loc == 'center':
        box_pts = boxes[:, :2] + boxes[:, 2:] / 2
    elif loc == 'bottom':
        box_pts = boxes[:, :2].copy()
        box_pts[:, 0] += boxes[:, 2] / 2 # x + w/2
        box_pts[:, 1] += boxes[:, 3]     # y + h
    return box_pts

def plot_loss(file_name, loss, use_log = True):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows, cols, size = 1, 1, 5
    font_size = 15
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
    if use_log:
        ax.plot(np.log(loss))
        ax.set_ylabel("Log(L1 loss)", fontsize = font_size)
    else:
        ax.plot(loss)
        ax.set_ylabel("L1 loss", fontsize = font_size)
    ax.set_xlabel("Epochs", fontsize = font_size)
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def plot_loss_v2(file_name, loss, loss_name = 'L1', use_log = True):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows, cols, size = 1, 1, 5
    font_size = 15
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
    if use_log:
        ax.plot(np.log(loss))
        ax.set_ylabel("Log({} loss)".format(loss_name), fontsize = font_size)
    else:
        ax.plot(loss)
        ax.set_ylabel("{} loss".format(loss_name), fontsize = font_size)
    ax.set_xlabel("Epochs", fontsize = font_size)
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def plot_loss_v3(file_name, loss, degrees, lower, upper, loss_name = 'velocity'):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    rows, cols, size = 1, 1, 5
    font_size = 15
    loss = np.clip(loss, lower, upper)
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
    ax.plot(degrees.flatten(), loss)
    ax.set_ylabel("{} loss".format(loss_name), fontsize = font_size)
    ax.set_xlabel("alpha", fontsize = font_size)
    ax.grid()
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def plot_1d_curve(file_name, ys, xs, 
                  ylabel = 'pair counts', 
                  xlabel = 'time window',
                  title  = 'without filtering',
                  ):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt
    rows, cols, size = 1, 1, 5
    font_size = 15
    # loss = np.clip(loss, lower, upper)
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows)); ax = fig.subplots(rows,cols)
    ax.plot(xs.flatten(), ys)
    ax.set_ylabel("{}".format(ylabel), fontsize = font_size)
    ax.set_xlabel("{}".format(xlabel), fontsize = font_size)
    ax.set_title(title)
    ax.grid()
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def plot_hist(file_name, distances, heights, range = [[-2, 2], [0, 4]], bins = (20, 20)):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import matplotlib.pyplot as plt
    rows, cols, size = 1, 1, 5
    font_size = 15
    fig = plt.figure(tight_layout=True,figsize=(size*cols, size*rows))
    ax = fig.subplots(rows,cols)
    plt.hist2d(heights, distances, range = range, bins= bins, cmap=plt.cm.jet)
    plt.colorbar()
    ax.set_xlabel("h_person/h_camera", fontsize = font_size)
    ax.set_ylabel("Distance", fontsize = font_size)
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

# load track by track id
def load_track_by_id(dataset_dir, track_id = 181, sample_interval = 3):
    track_names = os.listdir(dataset_dir)
    track = np.load(dataset_dir + '/{}.npy'.format(track_id))
    return track[::sample_interval,:]

def create_draw_board(w_img, h_img):
    return np.zeros((h_img, w_img, 3), dtype = np.uint8)

def create_white_board(w_img, h_img):
    return np.ones((h_img, w_img, 3), dtype = np.uint8) * 255

def lse_3d_flat(pts):
    pts = pts.reshape((-1, 3))
    num_pts = pts.shape[0]
    A = np.concatenate([pts, np.ones((num_pts, 1))], axis = -1)
    w, v = np.linalg.eig(np.matmul(A.T, A))
    print('eigen values:')
    print(w)
    min_index = np.argmin(w)
    print(w[min_index])
    return v[:, min_index]

def plot_surface_v2(file_name, xs, ys, zs,
                    z_clip = 0.,
                    xlabel = 'angles',
                    ylabel = 'f_scales',
                    zlabel = 'loss',
                    title  = 'seq',
                    rot    = (60, 45)):
    eps = 1e-8
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib import cm
    rows, cols, size = 1, 1, 5
    font_size = 15
    border_ratio = 0.1
    fig = Figure(tight_layout=True,figsize=(size *(border_ratio + 1) *cols, size *rows))
    ax = fig.add_subplot(projection='3d')
    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    zmin, zmax = zs.min(), zs.max()
    # if z_clip > 0: zmax = z_clip
    Y, X = np.meshgrid(ys, xs)
    # print(Y)
    # print(X)
    # print(zs)
    # print('2d energy function:')
    # print(zmin, zmax)
    # print(X.shape)
    # print(Y.shape)
    # print(zs.shape)
    surf = ax.plot_surface(Y, X, zs, 
                    rstride = 8, 
                    cstride = 8, 
                    # alpha = 0.1,
                    cmap=cm.coolwarm,
                    vmin=zmin, 
                    vmax=zmax)
    # surf = ax.plot_wireframe(Y, X, zs)
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.set_zlim([zmin, zmax])
    # ax.set_xlim([xmin, xmax])
    # ax.set_ylim([ymin, ymax])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # ax.view_init(140, 60)
    ax.view_init(rot[0], rot[1])
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)

def plot_surface(file_name, flat_3d, pts_3d):
    eps = 1e-8
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    rows, cols, size = 1, 1, 5
    font_size = 15
    fig = Figure(tight_layout=True,figsize=(size*cols, size*rows))
    # ax = fig.subplots(rows,cols)
    ax = fig.add_subplot(projection='3d')
    # plot the flat 
    a, b, c, d = flat_3d.flatten()
    xmin, xmax = pts_3d[:, 0].min(), pts_3d[:, 0].max()
    ymin, ymax = pts_3d[:, 1].min(), pts_3d[:, 1].max()
    zmin, zmax = pts_3d[:, 2].min(), pts_3d[:, 2].max()
    X = list(np.linspace(xmin, xmax, 100))
    Y = list(np.linspace(ymin, ymax, 100))
    X, Y = np.meshgrid(X, Y)
    Z = - (a * X + b * Y + d) / (c + eps)
    ax.plot_surface(X, Y, Z, rstride = 8, cstride = 8, alpha = 0.1)
    ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], marker = '+')
    zmin = min(zmin, Z.min())
    zmax = max(zmax, Z.max())
    ax.set_zlim([zmin, zmax])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(-140, 60)
    canvas = FigureCanvasAgg(fig); canvas.print_figure(file_name, dpi=100)