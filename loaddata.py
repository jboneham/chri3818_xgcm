import os
import numpy as np
from xarray import open_dataset, DataArray
from xgcm import Grid
from mds import rdmds
# import xarray as xr
# import xgcm as xg


def create_dataset(ncdir, *ncfiles):
    if ncdir[-1] != "/":
        ncd = ncdir + "/"
    else:
        ncd = ncdir
    ncfs = []
    for i, ncfile in enumerate(ncfiles):
        if ncfile[-3:] != ".nc":
            ncfs.append(ncfile + ".nc")
        else:
            ncfs.append(ncfile)
    dataset = open_dataset(ncd + "grid.nc")
    dataset = dataset.assign_attrs(
        grid=Grid(dataset,
                  coords={"X": {"center": "X", "outer": "Xp1"},
                          "Y": {"center": "Y", "outer": "Yp1"},
                          "Z": {"center": "Z", "outer": "Zp1", "left": "Zl", "right": "Zu"}},
                  periodic=None,
                  default_shifts={"Z": {"center": "outer"}},
                  fill_value=0)
    )
    dataset = dataset.assign(PHrefC=(("Z"), rdmds(ncd + "PHrefC").squeeze()))
    dataset = dataset.assign(RhoRef=(("Z"), rdmds(ncd + "RhoRef").squeeze()))
    for ncf in ncfs:
        ds = open_dataset(ncd + ncf).squeeze().drop_vars(["diag_levels", "iter"])
        dataset = dataset.merge(ds.rename_dims(get_zdims(ds)))
    return dataset


def get_zdims(dataset):
    zfrom = [i for i in dataset.dims if i[0] == "Z"]
    zto = [i[0] if i[1] == "m" else i[:2] for i in zfrom]
    return {zfrom[i]: zto[i] for i, _ in enumerate(zfrom)}


def get_psi_ext(dataset, T=slice(None), mask=True):
    vvel = dataset.VVEL.isel(T=T)
    if mask is True:
        mask = dataset.grid.interp(dataset.HFacS.sum(dim="X"), "Z", to='outer', boundary="extend")
    else:
        mask = True
    return (
        vvel                                                             # Start with meridional velocity field v
        .pipe(lambda x: (x * dataset.HFacS * dataset.drF).sum(dim="Z"))          # Integrate v vertically to get depth-integrated meridional velocity (V)
        .pipe(lambda x: x / (dataset.HFacS * dataset.drF).sum(dim="Z"))          # Divide V by the depth to get depth averaged meridional velocity (v_bar)
        .pipe(lambda x: x * vvel.where(dataset.HFacS == 0, other=1.))    # Set 3-D meridional velocity at each (x,y,z) to v_bar
        .transpose(*vvel.dims)                                           # Reorder dimensions (they get mixed up for some reason)
        .pipe(lambda x: (x * dataset.HFacS * dataset.dxG).sum(dim="X"))          # Integrate zonally
        .pipe(lambda x: dataset.grid.cumsum(x * dataset.drF, 'Z', to='outer',    # Cumulative depth integral to get an overturning streamfunction
                                        boundary='fill', fill_value=0.))
        .where(mask)                                                             # Apply mask
        .pipe(lambda x: 1E-6 * x)                                                # Convert to Sv
    )


def get_psi_bt(dataset, T=slice(None), mask=True):
    uvel = dataset.UVEL.isel(T=T)
    if mask is True:
        mask = dataset.grid.interp(dataset.HFacW.sum(dim="Z"), "Y", boundary="extend")
    else:
        mask = True
    return (
        dataset.UVEL                                                             # Start with meridional velocity field v
        .pipe(lambda x: (x * dataset.HFacW * dataset.drF).sum(dim="Z"))          # Integrate u vertically to get depth-integrated zonal velocity (U)
        .pipe(lambda x: dataset.grid.cumsum(-x * dataset.dyG, 'Y',               # Cumulatively integrate U meridionally to get an overturning streamfunction
                                            boundary='fill', fill_value=0.))
        .where(dataset.grid.interp(dataset.HFacW.sum(dim="Z"), "Y",              # Mask points not in ocean
                                   boundary="extend"))
        .pipe(lambda x: 1E-6 * x)
    )


def get_psi_moc(dataset, T=slice(None), mask=True):
    vvel = dataset.VVEL.isel(T=T)
    if mask is True:
        mask = dataset.grid.interp(dataset.HFacS.sum(dim="X"), 'Z', to="outer", boundary="extend")
    else:
        mask = True
    return (
        vvel
        .pipe(lambda x: (x * dataset.HFacS * dataset.dxG).sum(dim="X"))           # Integrate v zonally to get zonally-integrated meridional velocity (V)
        .pipe(lambda x: dataset.grid.cumsum(x * dataset.drF, 'Z', to='outer',     # Integrate V cumulatively in the vertical to get an overturning streamfunction
                                            boundary='fill', fill_value=0.))
        .where(mask)
        .pipe(lambda x: 1E-6 * x)
    )


def get_wind_curl(dataset):
    if "oceTAUY" in dataset.variables:
        dx_tauy = dataset.grid.diff(dataset.oceTAUY, "X", boundary='fill')
    else:
        dx_tauy = 0.
    if "oceTAUX" in dataset.variables:
        dy_taux = dataset.grid.diff(dataset.oceTAUX, "Y", boundary='fill')
    else:
        dy_taux = 0.
    return dx_tauy - dy_taux


def get_ekman_cell_fracs(dataset, d_Ek):
    rc = dataset.RC
    dz = dataset.drF
    fracs = np.zeros_like(rc)
    depth = 0
    i = 0
    while depth < d_Ek:
        depth += dz[i]
        fracs[i] = 1
        i += 1
    fracs[i - 1] -= (depth - d_Ek)/dz[i - 1]
    fracs = DataArray(data=fracs, dims="Z", coords={"Z": dataset.Z})
    return dataset.HFacW.where(lambda x: x < fracs, fracs)


def get_v_ekman(dataset, rho0=1025, d_Ek=50):
    L0_tau_bar = (dataset.dxC*dataset.oceTAUX.where(dataset.HFacW.isel(Z=0), 0.)).sum(dim="Xp1")
    A = (dataset.dxC*dataset.drF*dataset.HFacW).sum(dim=["Xp1","Z"])
    f = dataset.fCori.isel(X=0).squeeze()
    HFacEk = get_ekman_cell_fracs(dataset, d_Ek)
    HFacDeep = dataset.HFacW - HFacEk
    v_Deep = L0_tau_bar/(rho0 * f * A)
    v_Ek = -dataset.oceTAUX/(rho0 * f * d_Ek) + v_Deep
    v_Ek_corr = v_Ek*HFacEk + v_Deep*HFacDeep
    return v_Ek_corr.transpose("T", "Z", "Y", "Xp1")


def get_psi_ekman(dataset, T=slice(None), mask=True, rho0=1025, d_Ek=50):
    vvel = get_v_ekman(dataset, rho0=rho0, d_Ek=d_Ek)
    if mask is True:
        mask = dataset.grid.interp(dataset.HFacS.sum(dim="X"), 'Z', to="outer", boundary="extend")
    else:
        mask = True
    return (
        vvel
        .pipe(lambda x: (x * dataset.dxC).sum(dim="Xp1"))           # Integrate v zonally to get zonally-integrated meridional velocity (V)
        .pipe(lambda x: dataset.grid.cumsum(x * dataset.drF, 'Z', to='outer',     # Integrate V cumulatively in the vertical to get an overturning streamfunction
                                            boundary='fill', fill_value=0.))
        .pipe(lambda x: 1E-6 * x)
        .pipe(lambda x: dataset.grid.interp(x, 'Y', to='outer', boundary='fill', fill_value=0.))
        .where(mask)
    )


def get_dim_name(dataarray, firstletter):
    firstletter = firstletter.capitalize()
    assert firstletter in ['X', 'Y', 'Z', 'T']
    return [i for i in dataarray.dims if i[0] == firstletter][0]


def bound_east(hfac, iy, iz):
    view1d = hfac[iz, iy, :]
    nx = view1d.shape[0]
    hmax = view1d.max()
    if hmax == 0.:
        return [((iz, iy, -1), 0.)]
    else:
        pass
    view1d = view1d/hmax
    f_old = view1d[nx - 1]
    out = [((iz, iy, -1), f_old)]
    ix = nx - 2
    while view1d[ix + 1] <= view1d[ix]:
        f_new = view1d[ix]
        if f_new > f_old:
            out.append(((iz, iy, ix), f_new - f_old))
            f_old = f_new
        else:
            pass
        ix -= 1
    return out


def bound_west(hfac, iy, iz):
    view1d = hfac[iz, iy, :]
    hmax = view1d.max()
    if hmax == 0.:
        return [((iz, iy, 0), 0.)]
    else:
        pass
    view1d = view1d/hmax
    f_old = view1d[0]
    out = [((iz, iy, 0), f_old)]
    ix = 1
    while view1d[ix - 1] <= view1d[ix]:
        f_new = view1d[ix]
        if f_new > f_old:
            out.append(((iz, iy, ix), f_new - f_old))
            f_old = f_new
        else:
            pass
        ix += 1
    return out


def get_boundary(hfac, p, eastorwest='E'):
    if eastorwest[0] == 'E' or eastorwest[0] == 'e':
        bound_func = bound_east
    elif eastorwest[0] == 'W' or eastorwest[0] == 'w':
        bound_func = bound_west
    else:
        raise ValueError("eastorwest must be 'eastern' or 'western'")
    xdim, ydim, zdim = [get_dim_name(p, i) for i in ["x","y","z"]]
    if "T" in p.dims:
        p = p.transpose("T", zdim, ydim, xdim)
        nt, nz, ny, nx = p.shape
        pdata = p.data
    else:
        nt = 1
        p = p.transpose(zdim, ydim, xdim)
        nz, ny, nx = p.shape
        pdata = p.data[np.newaxis, :, :, :]
    p_out = np.zeros((nt, nz, ny))
    sl_weights = np.empty((nz, ny), dtype=object)
    for k in range(nz):
        for j in range(ny):
            sl_weights[k, j] = bound_func(hfac.data, j, k)
    for t in range(nt):
        pt = pdata
        for k in range(nz):
            for j in range(ny):
                sl_w = sl_weights[k, j]
                p_out[t, k, j] = 0.
                for sl, weight in sl_w:
                    tsl = [t] + [i for i in sl]
                    p_out[t, k, j] += pt[tuple(tsl)]*weight
    return p_out.squeeze()
