import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3

def plot_3d_motion(save_path, kinematic_tree, joints, title,
                   figsize=(10, 10), fps=120, radius=4):

    # --- Title wrapping (as you had) ---
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    # --- Data prep ---
    data = np.asarray(joints).copy().reshape(len(joints), -1, 3)  # ensure numpy
    fig = plt.figure(figsize=figsize)
    # ax = fig.add_subplot(111, projection='3d')   # ✅ use supported API
    ax = p3.Axes3D(fig)
    fig.add_axes(ax)

    # precompute mins/maxs and normalization (your logic)
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]
    frame_number = data.shape[0]

    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)
        return xz_plane  # return the artist if using blit

    def set_view_and_limits():
        # ✅ you MUST set limits/view AFTER any ax.clear()
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_ylim3d([0, radius])
        ax.set_zlim3d([0, radius])
        ax.view_init(elev=120, azim=-90)
        # private but still works; can be omitted if undesired
        ax._dist = 7.5
        ax.grid(False)
        fig.suptitle(title, fontsize=20)
        plt.axis('off')
        # hide ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    # Optional: draw one frame for debugging
    # set_view_and_limits(); fig.savefig("debug_first.png")

    def update(index):
        # ✅ robust: clear the whole 3D axis, then re-set everything
        ax.clear()
        set_view_and_limits()

        # plane centered on current root
        plane = plot_xzPlane(MINS[0] - trajec[index, 0],
                             MAXS[0] - trajec[index, 0],
                             0,
                             MINS[2] - trajec[index, 1],
                             MAXS[2] - trajec[index, 1])

        # trajectory so far
        artists = [plane]
        if index > 1:
            ln_traj, = ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                                 np.zeros_like(trajec[:index, 0]),
                                 trajec[:index, 1] - trajec[index, 1],
                                 linewidth=1.0, color='blue')
            artists.append(ln_traj)

        # skeleton chains
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            lw = 4.0 if i < 5 else 2.0
            ln_seg, = ax.plot3D(data[index, chain, 0],
                                data[index, chain, 1],
                                data[index, chain, 2],
                                linewidth=lw, color=color)
            artists.append(ln_seg)

        # If you later switch to blit=True, return artists
        return artists

    # blit=False is simplest for 3D; if you set blit=True, update must return artists
    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps,
                        repeat=False, blit=False)

    # Use an explicit writer to avoid backend surprises
    writer = FFMpegWriter(fps=fps, bitrate=1800)
    ani.save(save_path, writer=writer)
    plt.close(fig)


def plot_single_motion(joint, save_path, fps):
    from mogen.utils import paramUtil
    plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title='None', fps=fps, radius=4)


def plot_t2m(data, save_dir, captions, m_lengths, joints_num=22, fps=30, radius=4):
    import torch
    from os.path import join as pjoin
    from mogen.utils import paramUtil
    from mogen.utils.motion_representation import recover_from_ric

    tail = '.mp4'
    for i, (caption, joint_data) in enumerate(zip(captions, data)):
        joint_data = joint_data[:m_lengths[i]]
        joint = recover_from_ric(torch.from_numpy(joint_data).float(), joints_num).numpy()
        save_path = pjoin(save_dir, str(i).zfill(2) + tail)
        plot_3d_motion(save_path, paramUtil.t2m_kinematic_chain, joint, title=caption, fps=fps, radius=radius)