import warnings
import numpy as np
# try to import z5py
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
    """
    def __init__(self, path, key, io_module, voxel_size=None):
        self.path = path
        self.key = key
        self.ff = io_module.File(self.path)
        assert key in self.ff, "%s, %s" % (self.path, self.key)
        self.dataset = self.ff[key]
        try:
            self._voxel_size = tuple(np.array(self.dataset.attrs['resolution']).astype(np.int))
            if voxel_size is not None and voxel_size != self._voxel_size:
                warnings.warn("specified voxel size does not match voxel size saved in data")
        except KeyError:
            self._voxel_size = voxel_size
        assert self.voxel_size is not None
        self._shape_vc = self.dataset.shape
        self._shape = tuple(np.array(self._shape_vc) * np.array(self._voxel_size))

    def read(self, starts_wc, stops_wc):
        # make sure that things align with the voxel grid
        assert all(start_wc % res == 0 for start_wc, res in zip(starts_wc, self.voxel_size))
        assert all(stop_wc % res == 0 for stop_wc, res in zip(stops_wc, self.voxel_size))
        bb_vc = tuple(slice(start_wc/res, stop_wc/res) for start_wc, stop_wc, res in zip(starts_wc, stops_wc,
                                                                                   self._voxel_size))
        return self.read_vc(bb_vc)

    def read_vc(self, bounding_box_vc):
        return self.dataset[bounding_box_vc]

    def write_vc(self, out, out_bb_vc):
        # if isinstance(out, list):
        #     for ds, ch, o, bb in zip(self.datasets, self.channel_order, out, out_bb):
        #         assert o.ndim == 3
        #         print(ds[bb].shape, o.shape,bb, ds.shape)
        #         ds[bb] = o
        # else:
        #     for ds, ch in zip(self.datasets, self.channel_order):
        #         if isinstance(ch, list):
        #             assert out.ndim == 4
        #             # FIXME
        #             # z5py can't be called with a list as slicing index, hence this does not work.
        #             # this means, that we can only assign all channels to a single outputfile for now.
        #             # the best way to fix this would be to implement indexing by list in z5py
        #             # ds[(slice(None),) + out_bb] = out[ch]
        #             ds[(slice(None),) + out_bb] = out
        #         else:
        #             assert out[ch].ndim == 3
        #             ds[out_bb] = out[ch]
        assert out.ndim == len(self.shape)
        self.dataset[out_bb_vc] = out

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
            stops_wc = tuple([off_wc + outs * res for off_wc, outs, res in zip(offset_wc, arr.shape[1:],
                                                                             self.voxel_size)])
        else:
            stops_wc = tuple([off_wc + outs * res for off_wc, outs, res in zip(offset_wc, arr.shape,
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
    def __init__(self, path, keys,voxel_size=None):
        assert WITH_H5PY, "Need h5py"
        super(IoHDF5, self).__init__(path, keys, h5py, voxel_size=voxel_size)

    def close(self):
        self.ff.close()


class IoN5(IoBase):
    def __init__(self, path, keys, channel_order=None, voxel_size=None):
        assert WITH_Z5PY, "Need z5py"
        super(IoN5, self).__init__(path, keys, z5py, voxel_size=voxel_size)


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
