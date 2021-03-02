import os
import numpy as np
from xarray import open_dataset
from xgcm import Grid
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
