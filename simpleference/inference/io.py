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
    def __init__(self, path, keys, io_module, channel_order=None):
        assert isinstance(keys, (tuple, list)), type(keys)
        #assert len(keys) in (1, 2), str(len(keys))
        self.path = path
        self.keys = keys
        self.ff = io_module.File(self.path)
        assert all(kk in self.ff for kk in self.keys), "%s, %s" % (self.path, self.keys)
        self.datasets = [self.ff[kk] for kk in self.keys]
        # we just assume that everything has the same shape...
        self._shape = self.datasets[0].shape
        if channel_order is None:
            self.channel_order = list(range(len(self.keys)))
        else:
            self.channel_order = channel_order
        assert all(isinstance(ch, (int, list)) for ch in self.channel_order)

    def read(self, bounding_box):
        assert len(self.datasets) == 1
        return self.datasets[0][bounding_box]

    def write(self, out, out_bb):
        if isinstance(out,list):
            for ds, ch, o, bb in zip(self.datasets,self.channel_order, out, out_bb):
                assert o.ndim == 3
                print(ds[bb].shape, o.shape,bb, ds.shape)
                ds[bb] = o
        else:
            for ds, ch in zip(self.datasets, self.channel_order):
                if isinstance(ch, list):
                    assert out.ndim == 4
                    # FIXME
                    # z5py can't be called with a list as slicing index, hence this does not work.
                    # this means, that we can only assign all channels to a single outputfile for now.
                    # the best way to fix this would be to implement indexing by list in z5py
                    # ds[(slice(None),) + out_bb] = out[ch]
                    ds[(slice(None),) + out_bb] = out
                else:
                    assert out[ch].ndim == 3
                    ds[out_bb] = out[ch]

    @property
    def shape(self):
        return self._shape

    def close(self):
        pass


class IoHDF5(IoBase):
    def __init__(self, path, keys, channel_order=None):
        assert WITH_H5PY, "Need h5py"
        super(IoHDF5, self).__init__(path, keys, h5py, channel_order)

    def close(self):
        self.ff.close()


class IoN5(IoBase):
    def __init__(self, path, keys, channel_order=None):
        assert WITH_Z5PY, "Need z5py"
        super(IoN5, self).__init__(path, keys, z5py, channel_order)


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
