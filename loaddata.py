import os
import numpy as np
from xarray import open_dataset
from xgcm import Grid
# import xarray as xr
# import xgcm as xg


def get_diags(ncdir, diags3d, diags1d=None, levels=16):
    if ncdir[-1] != "/":
        ncdir = ncdir + "/"
    else:
        pass
    if diags3d[-3:] != ".nc":
        diags3d = diags3d + ".nc"
    else:
        pass
    if diags1d is not None and diags1d[-3:] != ".nc":
        diags1d = diags1d + ".nc"
    else:
        pass
    ds_grid = open_dataset(ncdir + "grid.nc")
    dim_dict = {f"Zmd{levels:0>6}":"Z", f"Zld{levels:0>6}":"Zl"}
    diags = (
        ds_grid.merge(open_dataset(ncdir + diags3d)
                      .drop_vars(["diag_levels", "iter"])
                      .rename_dims(**dim_dict)
                      .assign_coords({"Z" : ds_grid.Z, "Zl" : ds_grid.Zl,
                                      "Zu": ds_grid.Zu, "Zp1": ds_grid.Zp1}))
    )
    if diags1d is not None:
        diags = (diags.merge(open_dataset(ncdir + diags1d)
                             .squeeze()
                             .drop_vars(["diag_levels", "iter"])))
    else:
        pass
    diags = diags.assign_attrs(
        grid=Grid(diags,
                  coords={"X": {"center": "X", "outer": "Xp1"},
                          "Y": {"center": "Y", "outer": "Yp1"},
                          "Z": {"center": "Z", "outer": "Zp1", "left": "Zl", "right": "Zu"}},
                  periodic=None,
                  default_shifts={"Z": {"center": "outer"}},
                  fill_value=0)
    )
    return diags
