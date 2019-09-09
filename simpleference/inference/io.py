import warnings
import numpy as np
# try to import z5py
import z5py
try:
    import z5py
    WITH_Z5PY = True
except ImportError:
    WITH_Z5PY = False

# try to import h5py
try:
    import h5py
    WITH_H5PY = True
except ImportError:
    WITH_H5PY = False

# try to import dvid
try:
    from libdvid import DVIDNodeService
    from libdvid import ConnectionMethod
    WITH_DVID = True
except ImportError:
    WITH_DVID = False


class IoBase(object):
    """
    Base class for I/O with h5 and n5

    Arguments:
        path (str): path to h5 or n5 file
        keys (str or list[str]): key or list of keys to datasets in file
        io_module (io python module): needs to follow h5py syntax.
            either z5py or h5py
        channel_orders (list[slice]): mapping of channels to output datasets (default: None)
    """
    def __init__(self, path, keys, io_module, channel_order=None, voxel_size=None):
        assert isinstance(keys, (tuple, list, str)), type(keys)
        self.path = path
        self.keys = keys if isinstance(keys, (list, tuple)) else [keys]
        self.ff = io_module.File(self.path)
        assert all(kk in self.ff for kk in self.keys), "%s, %s" % (self.path, self.keys)
        self.datasets = [self.ff[kk] for kk in self.keys]
        # we just assume that everything has the same shape and voxel size...
        try:
            self._voxel_size = tuple(np.array(self.datasets[0].attrs['resolution']).astype(np.int))
            if voxel_size is not None and voxel_size != self._voxel_size:
                warnings.warn("specified voxel size does not match voxel size saved in data")
        except KeyError:
            self._voxel_size = voxel_size
        assert self._voxel_size is not None
        self._shape_vc = self.datasets[0].shape
        self._shape = tuple(np.array(self._shape_vc) * np.array(self._voxel_size))

        # validate non-trivial channel orders
        if channel_order is not None:
            assert all(isinstance(cho, slice) for cho in channel_order)
            assert len(channel_order) == len(self.datasets)
            for ds, ch in zip(self.datasets, channel_order):
                n_chan = ch.stop - ch.start
                if ds.ndim == 4:
                    assert n_chan == ds.shape[0]
                elif ds.ndim == 3:
                    assert n_chan == 1
                else:
                    raise RuntimeError("Invalid dataset dimensionality")
            self.channel_order = channel_order

        else:
            assert len(self.datasets) == 1, "Need channel order if given more than one dataset"
            self.channel_order = None

    def read(self, starts_wc, stops_wc):
        # make sure that things align with the voxel grid
        assert all(start_wc % res == 0 for start_wc, res in zip(starts_wc, self.voxel_size))
        assert all(stop_wc % res == 0 for stop_wc, res in zip(stops_wc, self.voxel_size))
        assert len(self.datasets) == 1
        bb_vc = tuple(slice(start_wc/res, stop_wc/res) for start_wc, stop_wc, res in zip(starts_wc, stops_wc,
                                                                                   self._voxel_size))
        return self.read_vc(bb_vc)


    def read_vc(self, bounding_box_vc):
        return self.datasets[0][bounding_box_vc]
    def write_vc(self, out, out_bb_vc):
        if self.channel_order is None:
            ds = self.datasets[0]
            assert out.ndim == ds.ndim, "%i, %i" % (out.ndim, ds.ndim)
            if out.ndim == 4:
                ds[(slice(None),) + out_bb_vc] = out
            else:
                ds[out_bb_vc] = out
        else:
            for ds, ch in zip(self.datasets, self.channel_order):
                if ds.ndim == 3:
                    ds[out_bb_vc] = out[ch][0]
                else:
                    ds[(slice(None),) + out_bb_vc] = out[ch]

    def write(self, out, offsets_wc):
        assert all(offset_wc % res == 0 for offset_wc, res in zip(offsets_wc, self.voxel_size))
        stops_wc = tuple([offset_wc + out_sh * res for offset_wc, out_sh, res in zip(offsets_wc,
                                                                                     out.shape, self.voxel_size)])
        assert all(stop_wc % res == 0 for stop_wc, res in zip(stops_wc, self.voxel_size))
        bb_vc = tuple(slice(start_wc/res, stop_wc/res) for start_wc, stop_wc, res in zip(offsets_wc, stops_wc,
                                                                                   self.voxel_size))
        return self.write_vc(out, bb_vc)

    def verify_block_shape(self, offset_wc, arr):
        if arr.ndim == 4:
            stops_wc = tuple([off_wc + out_sh * res for off_wc, out_sh, res in zip(offset_wc, arr.shape[1:],
                                                                             self.voxel_size)])
        else:
            stops_wc = tuple([off_wc + out_sh * res for off_wc, out_sh, res in zip(offset_wc, arr.shape,
                                                                               self.voxel_size)])

        # test whether block is overhanging, then crop
        if any(stop_wc > sh_wc for stop_wc, sh_wc in zip(stops_wc, self.shape)):
            arr_stops_wc = [sh_wc-off_wc if stop_wc > sh_wc else None
                            for stop_wc, sh_wc, off_wc in zip(stops_wc, self.shape, offset_wc)]
            assert all(arr_stop_wc%res == 0 if arr_stop_wc is not None else True
                       for arr_stop_wc, res in zip(arr_stops_wc, self.voxel_size))
            arr_stops_vc = [arr_stop_wc/res if arr_stop_wc is not None else None
                            for arr_stop_wc, res in zip(arr_stops_wc, self.voxel_size)]
            bb_vc = tuple(slice(0, arr_stop_vc) for arr_stop_vc in arr_stops_vc)
            if arr.ndim == 4:
                bb_vc = ((slice(None),) + bb_vc)
            arr = arr[bb_vc]
        return arr

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def shape(self):
        return self._shape

    def close(self):
        pass


class IoHDF5(IoBase):
    def __init__(self, path, keys, channel_order=None, voxel_size=None):
        assert WITH_H5PY, "Need h5py"
        super(IoHDF5, self).__init__(path, keys, h5py, channel_order=channel_order, voxel_size=voxel_size)

    def close(self):
        self.ff.close()


class IoN5(IoBase):
    def __init__(self, path, keys, channel_order=None, voxel_size=None):
        assert WITH_Z5PY, "Need z5py"
        super(IoN5, self).__init__(path, keys, z5py, channel_order=channel_order, voxel_size=voxel_size)


class IoDVID(object):
    def __init__(self, server_address, uuid, key):
        assert WITH_DVID, "Need dvid"
        self.ds = DVIDNodeService(server_address, uuid)
        self.key = key

        # get the shape the dvid way...
        endpoint = "/" + self.key + "/info"
        attributes = self.ds.custom_request(endpoint, "", ConnectionMethod.GET)
        # TODO do we need to increase by 1 here ?
        self._shape = tuple(mp + 1 for mp in attributes["MaxPoint"])

    def read(self, bb):
        offset = tuple(b.start for b in bb)
        shape = tuple(b.stop - b.start for b in bb)
        return self.ds.get_gray3D(self.key, shape, offset)

    def write(self, out, out_bb):
        raise NotImplementedError("Writing to DVID is not yet implemented!")

    @property
    def shape(self):
        return self._shape

    def close(self):
        pass
