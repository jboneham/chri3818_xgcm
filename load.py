import os
import re
import xarray as xr
import xgcm
import numpy as np


class mitgcmDataSet:

    def __init__(self, load_dir):
        self.load_dir = load_dir
        # self.data_files = os.listdir(load_dir)
        self.data_files = [i.decode("ascii") if type(i) == bytes else i for i in os.listdir(load_dir)]
        self.description = ""

    def load_grid(self):
        if "grid.nc" not in self.data_files:
            print(self.load_dir)
            print(self.data_files)
            print(self.load_dir)
            raise FileNotFoundError("no grid.nc file present in load directory")
        else:
            pass
        self.ds_grid = xr.open_dataset(self.load_dir + "/grid.nc")
        g = self.ds_grid.assign(drL=(("Zl"), self.ds_grid.drC.data[:-1]))
        g = g.assign(drU=(("Zu"), g.drC.data[1:]))
        crds = {'X': {'center': 'X', 'outer': 'Xp1'},
                'Y': {'center': 'Y', 'outer': 'Yp1'},
                'Z': {'center': 'Z', 'left': 'Zl', 'outer': 'Zp1', 'right': 'Zu'},
                'T': {'center': 'T'}}
        metrics = {
            ('X',): ['dxC', 'dxF', 'dxV', 'dxG'],
            ('Y',): ['dyC', 'dyF', 'dyU', 'dyG'],
            ('Z',): ['drF', 'drC', 'drL', 'drU'],
            ('X', 'Y'): ['rA', 'rAz', 'rAs', 'rAw']
        }
        self.grid = xgcm.Grid(
            g, coords=crds, periodic=["X", "Y"], metrics=metrics
        )
        self.mZeta = self.grid.interp(g.HFacC, ["X", "Y"], to="outer", boundary="extend")
        self.mUvel = g.HFacW
        self.mVvel = g.HFacS
        self.mWvel = self.grid.interp(g.HFacC, "Z", to="left", boundary="extend")
        self.mTrcr = g.HFacC
        self.mAmoc = self.grid.interp(g.HFacS, "Z", to="outer", boundary="extend")
        self.mAmoc = self.grid.average(self.mAmoc, "X")

    def avg(self, *args, **kwargs):
        return self.grid.average(*args, **kwargs)

    def intrp(self, *args, **kwargs):
        return self.grid.interp(*args, **kwargs)

    def intgrt(self, *args, **kwargs):
        return self.grid.integrate(*args, **kwargs)

    def cmsm(self, *args, **kwargs):
        return self.grid.cumsum(*args, **kwargs)

    def drvtv(self, *args, **kwargs):
        return self.grid.derivative(*args, **kwargs)

    def dff(self, *args, **kwargs):
        return self.grid.diff(*args, **kwargs)

    def cmnt(self, *args, **kwargs):
        return self.grid.cumint(*args, **kwargs)

    def load_nc_file(self, nc_file, ds_name):
        if hasattr(self, ds_name):
            print("There is already a dataset called {}".format(ds_name))
        else:
            if nc_file[-3:] != ".nc":
                nc_f = "{}.nc".format(nc_file)
            else:
                nc_f = nc_file
            if nc_f not in self.data_files:
                raise FileNotFoundError("no {} file present in load directory".format(nc_f))
            else:
                pass
            ds = xr.open_dataset("{}/{}".format(self.load_dir, nc_f))
            if hasattr(ds, "diag_levels"):
                ds = ds.drop_vars("diag_levels")
            else:
                pass
            for dim, size in ds.dims.items():
                if size == 1:
                    ds = ds.squeeze(dim=dim, drop=True)
                else:
                    pass
            rename = {"Zmd[0-9]+": "Z",
                    "Zld[0-9]+": "Zl",
                    "Zud[0-9]+": "Zu"}
            for key, val in rename.items():
                wrong_dim = [i for i in ds.dims if re.match(key, i)]
                if len(wrong_dim) > 0:
                    ds = ds.rename_dims({wrong_dim[0]: val})
                    ds = ds.update({val: self.ds_grid[val]})
                else:
                    pass
            setattr(self, ds_name, ds)

    def calc_all(self, ds_name):
        for f in [self.calc_psi, self.calc_psiOC, self.calc_div, self.calc_ke]:
            f(ds_name)

    def calc_psi(self, ds_name):
        ds = getattr(self, ds_name)
        g = self.grid
        uMass = (ds.UVEL * self.mUvel).fillna(0.)
        ds["psi"] = g.cumint(
            -g.integrate(uMass, 'Z'), 'Y', boundary='fill'
        )

    def calc_psiOC(self, ds_name):
        ds = getattr(self, ds_name)
        g = self.grid
        vMass = (ds.VVEL * self.mVvel).fillna(0.)
        ds["psiOC"] = g.cumint(
            g.integrate(vMass, 'X'), 'Z', to='outer', boundary='fill', fill_value=0.
        )
        # gds = self.ds_grid
        # ds["psiOC"] = g.cumsum(
        #     g.integrate(ds.VVEL, 'X') * gds.drF, 'Z', boundary='fill'
        # )
        # ds["psiOC"] = g.interp(ds["psiOC"], 'Z', boundary='fill')

    def calc_ke(self, ds_name):
        ds = getattr(self, ds_name)
        uMass = (ds.UVEL * self.mUvel).fillna(0.)
        vMass = (ds.VVEL * self.mVvel).fillna(0.)
        # gds = self.ds_grid
        g = self.grid
        u_sqr = g.interp(uMass**2, 'X', to='center')
        v_sqr = g.interp(vMass**2, 'Y', to='center')
        ds["ke"] = u_sqr + v_sqr

    def calc_div(self, ds_name):
        ds = getattr(self, ds_name)
        gds = self.ds_grid
        g = self.grid
        u_trans = (ds.UVEL * gds.dyG * gds.HFacW * gds.drF).fillna(0.)
        v_trans = (ds.VVEL * gds.dxG * gds.HFacS * gds.drF).fillna(0.)
        ds["div"] = (g.diff(u_trans, "X") + g.diff(v_trans, "Y"))/gds.rA


def load_run(exp_path="2008201654_ToS_01/run/init", nc_args=(("diagstate", "stt", True),)):
    base = "/network/group/aopp/oceans/DPM006_BONEHAM_NASPG/experiments/"
    ld = base + exp_path + "/STITCHED_OUTPUT/"
    print(f"> loading from {exp_path}")
    ds = mitgcmDataSet(ld)
    print("  >> initialised ds")
    ds.load_grid()
    print("  >> loaded grid")
    for nc, ns, stt_p in nc_args:
        print(f"  >> loading data from {nc} into namespace {ns}")
        ds.load_nc_file(nc, ns)
        print("    >>> done")
        if stt_p:
            print("  >> calculating additional diagnostics")
            # ds.calc_all("stt")
            print("    >>> calculating barotropic streamfunction")
            ds.calc_psi(ns)
            print("    >>> calculating overturning streamfunction")
            ds.calc_psiOC(ns)
            print("    >>> calculating divergence")
            ds.calc_div(ns)
            print("    >>> calculating kinetic energy")
            ds.calc_ke(ns)
            print("    >>> done")
        else:
            print("  >> no additional diagnostics calculated")
        print("  >> done")
    print("> done")
    return ds
