import numpy as np
from matplotlib import pyplot as plt
from matplotlib import tri
from shapely.geometry import Polygon, Point

from nonlinear_FEM_solid_2D import boundary_nodes

label = 'test'
letter_size = 12
plt.rc('font', family='Times New Roman', size=letter_size)


def plot_deformed_shapes(Nodes, Elements, Displacements, scale_factor=10., savefig=False):
    Displacements = Displacements.reshape(-1, 2)
    fig = plt.figure()
    for elem in Elements:
        node_ids = elem
        original_coords = Nodes[node_ids]
        disp = Displacements[node_ids]
        deformed_coords = original_coords + scale_factor * disp

        plt.plot(np.append(original_coords[:, 0], original_coords[0, 0]),
                 np.append(original_coords[:, 1], original_coords[0, 1]), 'r--')

        plt.plot(np.append(deformed_coords[:, 0], deformed_coords[0, 0]),
                 np.append(deformed_coords[:, 1], deformed_coords[0, 1]), 'b-')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Deformed Shapes of Elements (Magnification: {scale_factor}×)')
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    if savefig:
        fig.savefig(f'./figures/fig_deformed_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)


def plot_displacement(Nodes, Displacements, savefig=False):
    displacement_x = Displacements[::2]
    displacement_y = Displacements[1::2]

    X_plot = Nodes[:, 0]
    Y_plot = Nodes[:, 1]

    polygon = Polygon(boundary_nodes)

    def is_triangle_inside():
        inside_mask = np.ones(triangles.shape[0], dtype=bool)
        for i, tri_idx in enumerate(triangles):
            coord1 = X_plot[tri_idx]
            coord2 = Y_plot[tri_idx]
            centroid = Point(np.mean(coord1), np.mean(coord2))
            if polygon.contains(centroid):
                inside_mask[i] = False
        return inside_mask

    Tri = tri.Triangulation(X_plot, Y_plot)
    triangles = Tri.triangles
    mask = is_triangle_inside()
    Tri.set_mask(mask)

    # 绘制位移图
    fig = plt.figure(figsize=(11.5, 5))

    plt.subplot(1, 2, 1)
    plt.tricontourf(Tri, displacement_x, cmap='jet', levels=100)
    plt.colorbar()
    plt.axis('equal')
    plt.title(r'Displacement $\mathit{u}$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(1, 2, 2)
    plt.tricontourf(Tri, displacement_y, cmap='jet', levels=100)
    plt.colorbar()
    plt.axis('equal')
    plt.title(r'Displacement $\mathit{v}$')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.tight_layout()
    plt.show()

    if savefig:
        fig.savefig(f'./figures/fig_disp_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)


def plot_stress(Nodes, Elements, Stresses, savefig=False, figType='element'):
    sig_x = Stresses[:, 0]
    sig_y = Stresses[:, 1]
    tau_xy = Stresses[:, 2]

    x_coords = Nodes[:, 0]
    y_coords = Nodes[:, 1]
    times = np.zeros(Nodes.shape[0])
    s_xx = np.zeros(Nodes.shape[0])
    s_yy = np.zeros(Nodes.shape[0])
    s_xy = np.zeros(Nodes.shape[0])

    ele_flat = Elements.flatten()
    sig_x_expanded = np.repeat(sig_x, Elements.shape[1])
    sig_y_expanded = np.repeat(sig_y, Elements.shape[1])
    tau_xy_expanded = np.repeat(tau_xy, Elements.shape[1])
    np.add.at(s_xx, ele_flat, sig_x_expanded)
    np.add.at(s_yy, ele_flat, sig_y_expanded)
    np.add.at(s_xy, ele_flat, tau_xy_expanded)
    np.add.at(times, ele_flat, 1)
    s_xx = s_xx / times
    s_yy = s_yy / times
    s_xy = s_xy / times

    X_plot = x_coords.reshape(-1)
    Y_plot = y_coords.reshape(-1)
    sxx_plot = s_xx.reshape(-1)
    syy_plot = s_yy.reshape(-1)
    sxy_plot = s_xy.reshape(-1)

    # 绘制应力云图
    fig = plt.figure(figsize=(16.6, 5))
    if figType == 'node':
        polygon = Polygon(boundary_nodes)

        def is_triangle_inside():
            inside_mask = np.ones(triangles.shape[0], dtype=bool)
            for i, tri_idx in enumerate(triangles):
                coord1 = X_plot[tri_idx]
                coord2 = Y_plot[tri_idx]
                centroid = Point(np.mean(coord1), np.mean(coord2))
                if polygon.contains(centroid):
                    inside_mask[i] = False
            return inside_mask

        Tri = tri.Triangulation(X_plot, Y_plot)
        triangles = Tri.triangles
        mask = is_triangle_inside()
        Tri.set_mask(mask)

        plt.subplot(1, 3, 1)
        plt.tricontourf(Tri, sxx_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{xx}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.tricontourf(Tri, syy_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{yy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.tricontourf(Tri, sxy_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{xy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

    elif figType == 'element':
        plt.subplot(1, 3, 1)
        plt.tripcolor(x_coords, y_coords, Elements, facecolors=sig_x, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{xx}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.tripcolor(x_coords, y_coords, Elements, facecolors=sig_y, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\sigma_{yy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.tripcolor(x_coords, y_coords, Elements, facecolors=tau_xy, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\tau_{xy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError('figtype无效，请重新输入')

    if savefig:
        fig.savefig(f'./figures/fig_stress_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)


def plot_strain(Nodes, Elements, Strains, savefig=False, figtype='element'):
    exx_ele = Strains[:, 0]
    eyy_ele = Strains[:, 1]
    exy_ele = Strains[:, 2] * 0.5

    x_coords = Nodes[:, 0]
    y_coords = Nodes[:, 1]
    times = np.zeros(Nodes.shape[0])
    e_xx = np.zeros(Nodes.shape[0])
    e_yy = np.zeros(Nodes.shape[0])
    e_xy = np.zeros(Nodes.shape[0])

    ele_flat = Elements.flatten()
    exx_ele_expanded = np.repeat(exx_ele, Elements.shape[1])
    eyy_ele_expanded = np.repeat(eyy_ele, Elements.shape[1])
    exy_ele_expanded = np.repeat(exy_ele, Elements.shape[1])
    np.add.at(e_xx, ele_flat, exx_ele_expanded)
    np.add.at(e_yy, ele_flat, eyy_ele_expanded)
    np.add.at(e_xy, ele_flat, exy_ele_expanded)
    np.add.at(times, ele_flat, 1)
    e_xx = e_xx / times
    e_yy = e_yy / times
    e_xy = e_xy / times

    X_plot = x_coords.reshape(-1)
    Y_plot = y_coords.reshape(-1)
    exx_plot = e_xx.reshape(-1)
    eyy_plot = e_yy.reshape(-1)
    exy_plot = e_xy.reshape(-1)

    # 绘制应变云图
    fig = plt.figure(figsize=(16.9, 5))
    if figtype == 'node':
        polygon = Polygon(boundary_nodes)

        def is_triangle_inside():
            inside_mask = np.ones(triangles.shape[0], dtype=bool)
            for i, tri_idx in enumerate(triangles):
                coord1 = X_plot[tri_idx]
                coord2 = Y_plot[tri_idx]
                centroid = Point(np.mean(coord1), np.mean(coord2))
                if polygon.contains(centroid):
                    inside_mask[i] = False
            return inside_mask

        Tri = tri.Triangulation(X_plot, Y_plot)
        triangles = Tri.triangles
        mask = is_triangle_inside()
        Tri.set_mask(mask)

        plt.subplot(1, 3, 1)
        plt.tricontourf(Tri, exx_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{xx}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.tricontourf(Tri, eyy_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{yy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.tricontourf(Tri, exy_plot, cmap='jet', levels=100)
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{xy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

    elif figtype == 'element':
        plt.subplot(1, 3, 1)
        plt.tripcolor(x_coords, y_coords, Elements, facecolors=exx_ele, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{xx}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 2)
        plt.tripcolor(x_coords, y_coords, Elements, facecolors=eyy_ele, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{yy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 3, 3)
        plt.tripcolor(x_coords, y_coords, Elements, facecolors=exy_ele, cmap='jet')
        plt.colorbar()
        plt.axis('equal')
        plt.title(r'$\epsilon_{xy}$')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError('figtype无效，请重新输入')

    if savefig:
        fig.savefig(f'./figures/fig_strain_{label}.png', dpi=300, bbox_inches='tight', format='png', transparent=False)


if __name__ == '__main__':
    sol_path = f'./solutions/sol_{label}.npz'
    loaded_data = np.load(sol_path)
    node, element, displacement, stress, strain = (loaded_data['array1'], loaded_data['array2'], loaded_data['array3'],
                                                   loaded_data['array4'], loaded_data['array5'])
    # plot_deformed_shapes(node, element, displacement, scale_factor=0.1, savefig=False)  # scale_factor: 位移放大倍率
    plot_displacement(node, displacement, savefig=True)
    plot_stress(node, element, stress, savefig=True, figType='node')  # figType: 应力平滑策略（element: 无应力平滑；node: 第一类应力平滑）
    # plot_strain(node, element, strain, savefig=True, figtype='node')
