import numpy as np
import os
# import xmitgcm


_test_var = 1

_save_path = "/network/group/aopp/oceans/DPM006_BONEHAM_NASPG" + \
    "/MITgcm/defaults/input/"

_z_profile_default = np.array([
    [i] for i in [
        0., 50., 101., 153., 208., 276., 392., 567., 825., 1192., 1594.,
        2000., 2500., 3000., 3500., 4000.
    ]
])


def get_grid(lon_range, lat_range):
    """
    Generates a meshgrid given longitudes and latitudes
    ----------
    Parameters
    ----------
    lon_range: tuple/list
        Should be in format (min, max, delta) in degrees longitude
    lat_range: tuple/list
        Should be in format (min, max, delta) in degrees latitude
    ----------
    Side-Effects
    ----------
        None
    ----------
    Raises
    ----------
        None
    ----------
    Returns
    ----------
    tuple containing
    lon2: numpy.ndarray
        2-D array with the longitude of every point
    lat2: numpy.ndarray
        2-D array with the latitude of every point
    """
    lon = np.arange(lon_range[0], lon_range[1], lon_range[2])
    lat = np.arange(lat_range[0], lat_range[1], lat_range[2])
    lon2, lat2 = np.meshgrid(lon, lat)
    return lon2, lat2


def cos_lat(lon_range, lat_range):
    """
    Generates a meshgrid given longitudes and latitudes where the
    latitude steps are scaled by cos(lat)
    ----------
    Parameters
    ----------
    lon_range: tuple/list
        Should be in format (min, max, delta) in degrees longitude
    lat_range: tuple/list
        Should be in format (min, max, *doesn't matter*) in degrees latitude
    ----------
    Side-Effects
    ----------
        None
    ----------
    Raises
    ----------
        None
    ----------
    Returns
    ----------
    tuple containing
    lon2: numpy.ndarray
        2-D array with the longitude of every point
    lat2: numpy.ndarray
        2-D array with the latitude of every point
    """
    lon = np.arange(lon_range[0], lon_range[1], lon_range[2])
    deg2rad = 2*np.pi/360
    phi = deg2rad*lon_range[2]

    def solve_for_dtheta(theta, dtheta0):
        t_rad = deg2rad*theta
        dt_rad = deg2rad*dtheta0

        def f(dth, lat):
            return dth - (phi/2)*(np.cos(lat + dth) + np.cos(lat))

        def df(dth, lat):
            return 1 + (phi/2)*np.sin(lat + dth)

        n_iter = 0
        while f(dt_rad, t_rad) > 1E-9:
            if n_iter >= 20:
                raise RuntimeError("Not converging")
            else:
                dt_rad = dt_rad - f(dt_rad, t_rad)/df(dt_rad, t_rad)
                n_iter += 1
        return dt_rad/deg2rad

    dlat = [solve_for_dtheta(lat_range[0], lon_range[2])]
    n_iter = 0
    loop_lat = lat_range[0]
    lat = [loop_lat]
    while loop_lat < lat_range[1]:
        dlat.append(solve_for_dtheta(loop_lat, dlat[-1]))
        n_iter += 1
        loop_lat += dlat[-1]
        lat.append(loop_lat)
        if n_iter >= 300:
            raise RuntimeError("Too many latitude points required")
    if len(lat) % 2 == 0:
        lon2, lat2 = np.meshgrid(lon, lat)
    else:
        lon2, lat2 = np.meshgrid(lon, lat[:-1])
    return lon2, lat2, np.array(dlat)


"""
Topography
"""


def topog_flat(x_or_y_grid, h_0=-4000):
    """
    Generates a topography profile for a flat box, given a lat/long grid and
    maximum depth
    ----------
    Parameters
    ----------
    x_or_y_grid: numpy.ndarray
        Either of the two arrays returned by get_grid
    h_0: int
        The maximum depth (a negative number) of the profile
    ----------
    Side-Effects
    ----------
        None
    ----------
    Raises
    ----------
        None
    ----------
    Returns
    ----------
    h: numpy.ndarray
        2-D array of the depth over the domain defined by x_or_y_grid. It is
        flat-bottomed with depth h_0, but with zero-depth walls to suppress
        periodicity in MITgcm.
    """
    h = h_0*np.ones_like(x_or_y_grid)
    h[0, :] = 0
    h[-1, :] = 0
    h[:, 0] = 0
    h[:, -1] = 0
    return h


def topog_flat_nordic1_lab0(x_or_y_grid, h_0=-4000):
    n_x = np.shape(x_or_y_grid)[1]
    n_y = np.shape(x_or_y_grid)[0]
    n_x_lab = n_x//5
    n_x_nord = (3*n_x)//5
    n_y_nord = (2*n_y)//3 + 1
    h = topog_flat(x_or_y_grid, h_0)
    h[:, :n_x_lab] = 0
    h[n_y_nord:, :n_x_nord] = 0
    return h


def topog_flat_nordic0_lab0(xgrid, ygrid, h_0=-4000):
    # Gives the smallest row index where lat > 60 degrees (ie is outside basin)
    bound_N = np.argwhere(ygrid[:, 0] > 60)[0][0]
    # Gives the smallest col index where lon >= -50 degrees (just inside basin)
    bound_W = np.argwhere(xgrid[0, :] >= -50)[0][0]
    # Initialises depth to zero, then sets depth of area within bounds to h_0 #
    h = np.zeros_like(xgrid)
    h[:bound_N, bound_W:-1] = h_0
    # Sets one grid point's width minimum of land around basin ################
    h[0, :] = 0
    h[-1, :] = 0
    h[:, 0] = 0
    h[:, -1] = 0
    return h


def topog_flat_iceland1_lab0(x_grid, y_grid, h_0=-4000, sill=False):
    iceland = (np.argwhere(x_grid[0, :] >= -25)[0][0],  # west edge of Iceland
               np.argwhere(y_grid[:, 0] > 60)[0][0],  # south edge
               np.argwhere(x_grid[0, :] < -15)[-1][0],  # east edge
               np.argwhere(y_grid[:, 0] <= 65)[-1][0])  # north edge
    n_x_lab = np.argwhere(x_grid[0, :] >= -50)[0][0]
    n_x_nord = np.argwhere(x_grid[0, :] >= -30)[0][0]
    n_y_nord = np.argwhere(y_grid[:, 0] > 60)[0][0]
    h = topog_flat(y_grid, h_0)
    h[:, :n_x_lab] = 0
    h[n_y_nord:, :n_x_nord] = 0
    h[iceland[1]: iceland[3], iceland[0]: iceland[2]] = 0
    if not sill:
        pass
    elif sill == 1:
        h[n_y_nord: iceland[3], n_x_nord: iceland[0]] = -1000
        h[n_y_nord: iceland[3], iceland[2]: -1] = -1000
    elif sill == 2:
        i_1 = iceland[1]
        i_3 = iceland[3]
        for i in range(i_1, i_3):
            h[i, n_x_nord: iceland[0]] = -4000 + 3000*np.sin(
                np.pi*(i - i_1)/(i_3 - i_1)
            )
            h[i, iceland[2]: -1] = -4000 + 3000*np.sin(
                np.pi*(i - i_1)/(i_3 - i_1)
            )
        i_10 = i_1 + (i_3 - i_1)/3
        i_30 = i_1 + 2*(i_3 - i_1)/3
        print(i_1, i_10, i_3, i_30)
    elif sill == 3:
        i_1 = iceland[1]
        i_3 = iceland[3]
        delta = int(abs((i_3 - i_1)/2))
        delta = int(min(delta, (i_3 - i_1)/2))
        i_10 = i_1 + delta
        i_30 = i_3 - delta
        print(i_10, i_30, delta)
        func1 = (
            lambda x, x0, x1:
            (np.tanh(4*(x - x0 + 1)/abs(x1 - x0) - 2)/np.tanh(2) + 1)/2
        )
        func3 = (
            lambda x, x0, x1:
            (np.tanh(4*(x0 - x)/abs(x1 - x0) + 2)/np.tanh(2) + 1)/2
        )
        for i in range(i_1, i_10):
            h[i, n_x_nord: iceland[0]] = -4000 + 3000*func1(i, i_1, i_10)
            h[i, iceland[2]: -1] = -4000 + 3000*func1(i, i_1, i_10)
        for i in range(i_10, i_30):
            h[i, n_x_nord: iceland[0]] = -4000 + 3000
            h[i, iceland[2]: -1] = -4000 + 3000
        for i in range(i_30, i_3):
            h[i, n_x_nord: iceland[0]] = -4000 + 3000*func3(i, i_30, i_3)
            h[i, iceland[2]: -1] = -4000 + 3000*func3(i, i_30, i_3)
    else:
        pass
    return h


###
#  Wind Forcing
###

def tau_full(y_grid, tau_0=0.1):
    ny, nx = np.shape(y_grid)
    x = np.array([(i + 0.5)/(nx - 1) for i in range(nx)])
    y = np.array([(i + 0.5)/(ny - 1) for i in range(ny)])
    X, Y = np.meshgrid(x, y)
    tau = np.sin(np.pi*Y)*tau_0
    return tau


def tau_ocean_only(y_grid, tau_0=0.1):
    i_nordic = np.argwhere(y_grid[:, 0] > 60)[0][0]
    ny, nx = np.shape(y_grid)
    x = np.array([(i + 0.5)/(nx - 1) for i in range(nx)])
    y = np.array([(i + 0.5)/(i_nordic - 1) for i in range(i_nordic)])
    X, Y = np.meshgrid(x, y)
    tau = np.zeros_like(y_grid)
    tau[:i_nordic, :] = np.sin(np.pi*Y)*tau_0
    return tau


def tau_ocean_only_smoothed(y_grid, tau_0=0.1):
    i_lim = np.argwhere(y_grid[:, 0] >= 70)[0][0]
    ny, nx = np.shape(y_grid)
    x = np.array([(i + 0.5)/(nx - 1) for i in range(nx)])
    y = np.array([(i + 0.5)/(i_lim - 1) for i in range(i_lim)])
    X, Y = np.meshgrid(x, y)

    def smoothf(x):
        if x <= 0.5:
            y = np.sin(np.pi*x)
        else:
            y = np.sin(np.pi*x) + 1.39272*(x - 0.5)**2.356194
        return y

    fvec = np.vectorize(smoothf)
    tau = np.zeros_like(y_grid)
    tau[:i_lim, :] = fvec(1.25*Y)*tau_0
    return tau


def tau_sin_window(y_grid, tau_0=0.1, window=(20.0, 60.0)):
    imin = np.argwhere(y_grid[:, 0] > window[0])[0][0]
    imax = np.argwhere(y_grid[:, 0] >= window[1])[0][0]
    y = (
        (y_grid[imin:imax, 0] - y_grid[imin, 0])
        / (y_grid[imax - 1, 0] - y_grid[imin, 0])
    )
    x = y_grid[0, :]
    X, Y = np.meshgrid(x, y)
    tau = np.zeros_like(y_grid)
    tau[imin:imax, :] += np.sin(np.pi*Y)*np.sin(np.pi*Y)*tau_0
    return tau


###############################################################################
#                                 Temp Forcing                                #
###############################################################################


def temp_2_sponge(y_grid, temp_min=5, temp_max=20):
    n_y = np.shape(y_grid)[0]
    n_y_sponge_s = n_y//6 + 1
    n_y_sponge_n = (5*n_y)//6
    y = y_grid[n_y_sponge_s:n_y_sponge_n, :]
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    temp = np.ones_like(y_grid)
    temp[n_y_sponge_s:n_y_sponge_n, :] = temp_max - (temp_max - temp_min)*y
    temp[:n_y_sponge_s, :] = temp_max
    temp[n_y_sponge_n:, :] = temp_min
    return temp


def temp_constant_nordic(y_grid, temp_min=5, temp_max=20):
    n_y_sponge_s = np.argwhere(y_grid[:, 0] < 30)[-1, 0]
    n_y_sponge_n = np.argwhere(y_grid[:, 0] > 60)[0, 0]
    y = y_grid[n_y_sponge_s:n_y_sponge_n, :]
    y = (y - np.min(y))/(np.max(y) - np.min(y))
    temp = np.ones_like(y_grid)
    temp[n_y_sponge_s:n_y_sponge_n, :] = temp_max - (temp_max - temp_min)*y
    temp[:n_y_sponge_s, :] = temp_max
    temp[n_y_sponge_n:, :] = temp_min
    return temp


"""
Temperature Initialisation
"""


def temp_profile_cold_bw(
        lvl_5deg=9, bw_temp=1.,
        z_interface=np.array([
            0., 50., 101., 153., 208., 276., 392., 567., 825., 1192., 1594.,
            2000., 2500., 3000., 3500., 4000.
        ])
):
    """
    Note that this function returns a 1-D temperature profile to be manually
    entered into the data file
    """
    def newton():
        if type(lvl_5deg) == str:
            alpha = float(lvl_5deg)
        else:
            alpha = z_interface[lvl_5deg]
        F = lambda x: np.array([
            x[0] + x[2] - 20,
            x[0]*np.exp(-3500*x[1]) + x[2] - bw_temp,
            x[0]*np.exp(-alpha*x[1]) + x[2] - 5
        ])
        J = lambda x: np.vstack((
            np.array([1, 0, 1]),
            np.array([np.exp(-3500*x[1,0]), -3500*x[0,0]*np.exp(-3500*x[1,0]), 1]),
            np.array([np.exp(-alpha*x[1,0]), -alpha*x[0,0]*np.exp(-alpha*x[1,0]), 1])
        ))
        X = np.transpose(np.array([20, np.log(4)/alpha, 0])[np.newaxis])
        for i in range(10):
            X = X - np.matmul(np.linalg.inv(J(X)), F(X))
        return X

    C = newton()
    f = lambda x: C[0]*np.exp(-x*C[1]) + C[2]
    T_profile = f(z_interface)
    T_profile[-1] = T_profile[-2]
    return T_profile


def temp_init_merid_linear(
        y_grid, z_profile_N=np.array([5. for i in range(15)]),
        z_profile_S=np.array([
            20.00, 17.13, 14.77, 12.83, 11.25, 9.94,  8.88,  8.00,  7.28,
            6.69, 6.21,  5.81,  5.48,  5.22,  5.00
        ]), sponge_S_lat=30., sponge_N_lat=70.
):
    zpS = np.stack(
        [z_profile_S for i in range(np.shape(y_grid)[1])]).T
    zpN = np.stack(
        [z_profile_N for i in range(np.shape(y_grid)[1])]).T
    temp = np.zeros((np.shape(zpS)[0], np.shape(y_grid)[0], np.shape(zpS)[1]))
    dy = y_grid[1, 0] - y_grid[0, 0]
    dlat = sponge_N_lat - sponge_S_lat
    dT = zpS - zpN
    for i in range(np.shape(y_grid)[0]):
        if y_grid[i, 0] < sponge_S_lat:
            temp[:, i, :] = zpS
            i0 = i + 1
        elif y_grid[i, 0] >= sponge_N_lat:
            temp[:, i, :] = zpN
        else:
            temp[:, i, :] = zpS - dT*(i - i0)*dy/(dlat - dy)
    return temp


"""
Sponge Arrays
"""


def sponge_mask(y_grid, zdim=16):
    msk = np.zeros((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
    j_30 = np.argwhere(y_grid[:, 0] > 30)[0][0]
    j_70 = np.shape(y_grid)[0] - np.argwhere(y_grid[:, 0] >= 70)[0][0]
    for j, lat in enumerate(y_grid[:, 0]):
        if lat <= 30:
            print(j)
            msk[:, j, :] = (
                np.cos(np.pi*j/j_30) + 1)/2
        elif lat >= 70:
            msk[:, j, :] = (
                np.cos(np.pi*j/j_70) + 1)/2
    return msk


def sponge_mask_N1_S1_tanh(y_grid, zdim=16, scale=6, N_S_factor=1):
    msk = np.zeros((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
    j_30 = np.argwhere(y_grid[:, 0] > 30)[0][0] - 1
    j_70_0 = np.argwhere(y_grid[:, 0] >= 70)[0][0]
    j_70 = np.shape(y_grid)[0] - j_70_0 - 1
    for j, lat in enumerate(y_grid[:, 0]):
        if lat <= 30:
            msk[:, j, :] = N_S_factor*(
                np.tanh(scale/2) + np.tanh(scale/2 - scale*j/j_30)
                )
        elif lat >= 70:
            msk[:, j, :] = (
                np.tanh(scale/2) - np.tanh(scale/2 - scale*(j - j_70_0)/j_70)
            )
    msk = msk/np.max(msk)
    return msk


def sponge_mask_const_iceland(
        x_grid, y_grid, zdim=16, scale=6, N_S_factor=1, width="full"
):
    if width == "full":
        i_sponge = (np.argwhere(x_grid[0, :] >= -25)[0][0],
                    np.argwhere(x_grid[0, :] < -15)[-1][0])
    else:
        i_sponge = (np.argwhere(x_grid[0, :] >= -22.5)[0][0],
                    np.argwhere(x_grid[0, :] < -17.5)[-1][0])
    msk = np.zeros((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
    j_30 = np.argwhere(y_grid[:, 0] > 30)[0][0] - 1
    j_70_0 = np.argwhere(y_grid[:, 0] >= 70)[0][0]
    j_70 = np.shape(y_grid)[0] - j_70_0 - 1
    for j, lat in enumerate(y_grid[:, 0]):
        if lat <= 30:
            msk[:, j, :] = (
                np.tanh(scale/2) + np.tanh(scale/2 - scale*j/j_30)
                )
        elif lat >= 70:
            msk[:, j, i_sponge[0]: i_sponge[1]] = np.max(msk)
    msk = msk/np.max(msk)
    return msk


def sponge_mask_shallow_iceland(
        x_grid, y_grid, zdim=16, scale=6, N_S_factor=1, width="full", sponge_lvl=9
):
    msk = sponge_mask_const_iceland(
        x_grid, y_grid, zdim=16, scale=6, N_S_factor=1, width="full"
    )
    for j, lat in enumerate(y_grid[:, 0]):
        if lat >= 70:
            msk[sponge_lvl + 1:, j, :] = 0
    return msk


def sponge_mask_sin_iceland(
        x_grid, y_grid, zdim=16, scale=6, N_S_factor=1
):
    msk = np.zeros((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
    j_30 = np.argwhere(y_grid[:, 0] > 30)[0][0] - 1
    j_70_0 = np.argwhere(y_grid[:, 0] >= 70)[0][0]
    j_70 = np.shape(y_grid)[0] - j_70_0 - 1
    for j, lat in enumerate(y_grid[:, 0]):
        if lat <= 30:
            msk[:, j, :] = 0.5*(
                np.tanh(scale/2) + np.tanh(scale/2 - scale*j/j_30)
                )/(N_S_factor*0.995)
        elif lat >= 70:
            for i, lon in enumerate(x_grid[0, :]):
                if np.abs(lon + 20) < 5:
                    msk[:, j, i] = 0.5*(1 + np.cos(2*np.pi*(lon + 20)/10))
    return msk


# def old_sponge_mask_sin_iceland_unity(
#         x_grid, y_grid, tau_N, tau_S, zdim=16, scale=6
# ):
#     msk = np.zeros((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
#     j_30 = np.argwhere(y_grid[:, 0] > 30)[0][0] - 1
#     j_70_0 = np.argwhere(y_grid[:, 0] >= 70)[0][0]
#     j_70 = np.shape(y_grid)[0] - j_70_0 - 1
#     for j, lat in enumerate(y_grid[:, 0]):
#         if lat <= 30:
#             msk[:, j, :] = 0.5*(
#                 np.tanh(scale/2) + np.tanh(scale/2 - scale*j/j_30)
#                 )/(tau_S*0.995)
#         elif lat >= 70:
#             for i, lon in enumerate(x_grid[0, :]):
#                 width = 2
# # width = 2 => sponge across width of Iceland,
# # width = 1 => sponge across width of Nordic Sea
#                 if np.abs(lon + 20) < 10/width:
#                     msk[:, j, i] = 0.5*(1 + np.cos(width*np.pi*(lon + 20)/10))/tau_N
#     return msk


def sponge_mask_sin_iceland_unity(
        x_grid, y_grid, tau_S, tau_N, zdim=16, scale=6
):
    dims = (zdim, np.shape(y_grid)[0], np.shape(y_grid)[1])
    msk = np.zeros(dims)
    sponge_lat_S = 30
    sponge_lat_N = 65
    iceland_W = -25
    iceland_E = -15
    ###########################################################################
    # adding " + 1" to i_bound below means that the sponge extends out of the #
    # subpolar region by a grid point, but the relaxation at that point is    #
    # zero. This ensures that the gradient of relaxation timescale is         #
    # continuous across the sponge boundary                                   #
    ###########################################################################
    i_lbound = np.argwhere(y_grid[:, 0] <= sponge_lat_S)[-1][0] + 1
    i_ubound = np.argwhere(y_grid[:, 0] >= sponge_lat_N)[0][0]
    j_lbound = np.argwhere(x_grid[0, :] >= iceland_W)[0][0] - 1
    j_ubound = np.argwhere(x_grid[0, :] <= iceland_E)[-1][0] + 1
    j_width = j_ubound - j_lbound
    for i, lat in enumerate(y_grid[:, 0]):
        if i <= i_lbound:
            msk[:, i, :] = 0.5*(
                np.tanh(scale/2) + np.tanh(scale/2 - scale*i/i_lbound)
                )
        elif i >= i_ubound:
            for j, lon in enumerate(x_grid[0, :]):
                if iceland_W <= lon <= iceland_E:
                    msk[:, i, j] = (
                        0.5*(1 + np.sin(-np.pi/2 + 2*np.pi*(j - j_lbound)/j_width))
                    )
        else:
            pass
    msk[:, :i_lbound + 1, :] = (
        msk[:, :i_lbound + 1, :] / (tau_S * np.max(msk[:, :i_lbound + 1, :]))
    )
    msk[:, i_ubound:, :] = (
        msk[:, i_ubound:, :] / (tau_N * np.max(msk[:, i_ubound:, :]))
    )
    return msk


def sponge_mask_N0_S1_tanh(y_grid, zdim=16, scale=6):
    msk = np.zeros((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
    j_30 = np.argwhere(y_grid[:, 0] > 30)[0][0] - 1
    for j, lat in enumerate(y_grid[:, 0]):
        if lat <= 30:
            msk[:, j, :] = (
                np.tanh(scale/2) + np.tanh(scale/2 - scale*j/j_30)
                )
    msk = msk/np.max(msk)
    return msk


def sponge_temp_N1_S1(y_grid, zdim=16, profiles="default"):
    temp = np.ones((zdim, np.shape(y_grid)[0], np.shape(y_grid)[1]))
    """Profiles should NEVER be default!!!"""
    if profiles == "default":
        temp_N = 5.*np.ones(zdim)
        temp_S = np.array([
            20.00, 17.13, 14.77, 12.83, 11.25, 9.94,  8.88,  8.00,  7.28,
            6.69, 6.21,  5.81,  5.48,  5.22,  5.00,  5.00
        ])
    else:
        temp_N = profiles[0]
        temp_S = profiles[1]
        assert ((type(temp_N) == list) or (type(temp_N) == tuple) or
                (type(temp_N) == np.ndarray))
        assert ((type(temp_S) == list) or (type(temp_S) == tuple) or
                (type(temp_S) == np.ndarray))
        temp_N = np.array(temp_N)
        temp_S = np.array(temp_S)
    if (len(temp_N) != zdim) or (len(temp_S) != zdim):
        raise ValueError("Profile size not equal to depth dimension")
    else:
        pass
    temp_N = np.reshape(temp_N, (zdim, 1, 1))
    temp_S = np.reshape(temp_S, (zdim, 1, 1))
    i_split = np.argwhere(y_grid[:, 0] >= 50)[0][0]
    temp[:, :i_split, :] = temp[:, :i_split, :]*temp_S
    temp[:, i_split:, :] = temp[:, i_split:, :]*temp_N
    return temp


###############################################################################
#                            Perturbation Functions                           #
###############################################################################


def pert_sin2sin2_greatcircle(x, y, x_0, y_0, r_max):
    dx = np.abs(x - x_0)*np.pi/180
    dy = np.abs(y - y_0)*np.pi/180
    r_max = r_max*np.pi/180
    asin = np.arcsin
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    r = 2*asin(sqrt(sin(dy/2)**2 + cos(y*np.pi/180)*cos(y_0*np.pi/180)*sin(dx/2)**2))
    if r <= r_max:
        f_dash = cos(0.5*np.pi*r/r_max)**2
    else:
        f_dash = 0
    return f_dash


def pert_sin2sin2(x, y, x_0, y_0, r_max):
    dx = x - x_0
    dy = y - y_0
    r = np.sqrt(dx**2 + dy**2)
    if r <= r_max:
        f_dash = np.cos(0.5*np.pi*dx/r_max)**2 * np.cos(0.5*np.pi*dy/r_max)**2
        f_dash = np.cos(0.5*np.pi*r/r_max)**2
    else:
        f_dash = 0
    return f_dash


def apply_pert(x_grid, y_grid, f_0, pert_xy, pert_func, pert_amp, pert_rad):
    f_dash = np.zeros_like(f_0)
    dims = np.shape(f_dash)
    x_0 = pert_xy[0]
    y_0 = pert_xy[1]
    for i in range(dims[0]):
        for j in range(dims[1]):
            x = x_grid[i, j]
            y = y_grid[i, j]
            f_dash[i, j] = pert_amp * pert_func(x, y, x_0, y_0, pert_rad)
    return f_0 + f_dash


"""
File Writing
"""


# def write_to_binary(array, fname, input_dir=_save_path, precision='float64'):
#     savename = input_dir + fname
#     xmitgcm.utils.write_to_binary(array.flatten(), savename, dtype=precision)
#     return 0


# def specific_write(array, fname, exp, dtype="float64", **kwargs):
#     sp = "/network/group/aopp/oceans/DPM006_BONEHAM_NASPG/experiments/"
#     while exp not in os.listdir(sp):
#         print("experiment not found\nAvailable experiments are:")
#         [print(i) for i in os.listdir(sp)]
#         exp = input("Enter experiment >>>  ")
#     savename = sp + exp + "/input/" + fname + ".bin"
#     if os.path.exists(savename):
#         msg = "Overwrote "
#         os.remove(savename)
#     else:
#         msg = "Wrote "
#     print(msg + savename)
#     xmitgcm.utils.write_to_binary(array.flatten(), savename, dtype=dtype, **kwargs)
#     return 0


def write_all(
        h_0=-4000, tau_0=0.1, temp_min=5, temp_max=20, zdim=16, sponge_scale=6,
        prec='float64', T_profs="default",
        z_profile_N=np.array([5. for i in range(15)]),
        z_profile_S=np.array([
            20.00, 17.13, 14.77, 12.83, 11.25, 9.94,  8.88,  8.00,  7.28,
            6.69, 6.21,  5.81,  5.48,  5.22,  5.00
        ]), sponge_S_lat=30., sponge_N_lat=70.
):
    files = ()
    lon_range = (-60, -10, 0.5)
    lat_range = (20, 80, 0.5)
    x, y = get_grid(lon_range, lat_range)
    suffix = "_" + str(np.shape(x)[1]) + "x" + str(np.shape(x)[0]) + ".bin"
    files = files + \
        tuple(
            map(lambda x: (x[0], x[1]+suffix),
                (
                    (topog_flat(y, h_0),
                     "topog_flat_full_grid"),
                    (topog_flat_nordic1_lab0(y, h_0),
                     "topog_flat_nordic1_lab0"),
                    (topog_flat_nordic0_lab0(y, h_0),
                     "topog_flat_nordic0_lab0"),
                    (tau_full(y, tau_0),
                     "tau_full"),
                    (temp_2_sponge(y, temp_min, temp_max),
                     "temp_2_sponge"),
                    (sponge_mask_N0_S1_tanh(y, zdim=zdim, scale=sponge_scale),
                     "sponge_mask_N0_S1_tanh"),
                    (sponge_mask_N1_S1_tanh(y, zdim=zdim, scale=sponge_scale),
                     "sponge_mask_N1_S1_tanh"),
                    (sponge_mask_N1_S1_tanh(
                        y, zdim=zdim, scale=sponge_scale, N_S_factor=0.1
                    ),
                     "sponge_mask_N1_S1_tanh_NS_asym"),
                    (temp_init_merid_linear(
                        y, z_profile_N=z_profile_N, z_profile_S=z_profile_S,
                        sponge_S_lat=sponge_S_lat, sponge_N_lat=sponge_N_lat
                    ),
                     "temp_init_merid_linear"),
                    (sponge_temp_N1_S1(y, zdim=zdim, profiles=T_profs),
                     "sponge_temp_N1_S1"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[5.*np.ones(15),temp_profile_cold_bw(9,1)]
                    ),
                     "sponge_temp_cold_bw_9_1"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[5.*np.ones(15),temp_profile_cold_bw(9,3)]
                    ),
                     "sponge_temp_cold_bw_9_3"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[5.*np.ones(15),temp_profile_cold_bw(11,3)]
                    ),
                     "sponge_temp_cold_bw_11_3"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[5.*np.ones(15),temp_profile_cold_bw(10,1)]
                    ),
                     "sponge_temp_cold_bw_10_1"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[
                            np.minimum(5.*np.ones(15),temp_profile_cold_bw(10,1)),temp_profile_cold_bw(10,1)
                        ]
                    ),
                     "sponge_temp_cold_bw_taper_10_1"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[
                            np.minimum(5.*np.ones(15),temp_profile_cold_bw(10,2)),temp_profile_cold_bw(10,2)
                        ]
                    ),
                     "sponge_temp_cold_bw_taper_10_2"),
                    (sponge_temp_N1_S1(
                        y, zdim=zdim, profiles=[
                            np.minimum(5.*np.ones(15),temp_profile_cold_bw(10,3)),temp_profile_cold_bw(10,3)
                        ]
                    ),
                     "sponge_temp_cold_bw_taper_10_3"),
                    (topog_flat_iceland1_lab0(x, y, h_0=h_0),
                     "topog_flat_iceland1_lab0"),
                    (sponge_mask_const_iceland(
                        x, y, zdim=16, scale=6, N_S_factor=0.1, width="full"
                    ),
                     "sponge_mask_const_iceland_full"),
                    (sponge_mask_const_iceland(
                        x, y, zdim=16, scale=6, N_S_factor=0.1, width="narrow"
                    ),
                     "sponge_mask_const_iceland_narrow"),
                    (sponge_mask_sin_iceland(
                        x, y, zdim=16, scale=6, N_S_factor=10
                    ),
                     "sponge_mask_sin_iceland_x10"),
                    (sponge_mask_sin_iceland(
                        x, y, zdim=16, scale=6, N_S_factor=100
                    ),
                     "sponge_mask_sin_iceland_x100"),
                    (sponge_mask_sin_iceland(
                        x, y, zdim=16, scale=6, N_S_factor=1000
                    ),
                     "sponge_mask_sin_iceland_x1000"),
                    (sponge_mask_shallow_iceland(
                        x, y, zdim=16, scale=6, N_S_factor=0.1, width="narrow", sponge_lvl=11
                    ),
                     "sponge_mask_shallow_iceland_11")
                )))
    for f in files:
        print("writing {}".format(f[1]))
        write_to_binary(f[0], f[1])
    np.savetxt(
        _save_path + "cold_bw_profile_9_1.txt", temp_profile_cold_bw(9,1), fmt="%.2f",
        delimiter=", ", newline=", "
    )
    np.savetxt(
        _save_path + "cold_bw_profile_9_3.txt", temp_profile_cold_bw(9,3), fmt="%.2f",
        delimiter=", ", newline=", "
    )
    np.savetxt(
        _save_path + "cold_bw_profile_11_3.txt", temp_profile_cold_bw(11,3), fmt="%.2f",
        delimiter=", ", newline=", "
    )
    np.savetxt(
        _save_path + "cold_bw_profile_10_3.txt", temp_profile_cold_bw(10,3), fmt="%.2f",
        delimiter=", ", newline=", "
    )
    np.savetxt(
        _save_path + "cold_bw_profile_10_2.txt", temp_profile_cold_bw(10,2), fmt="%.2f",
        delimiter=", ", newline=", "
    )
    np.savetxt(
        _save_path + "cold_bw_profile_10_1.txt", temp_profile_cold_bw(10,1), fmt="%.2f",
        delimiter=", ", newline=", "
    )
    return 0
