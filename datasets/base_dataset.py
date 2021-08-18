try:
    import mc
except ModuleNotFoundError:
    print('import mc failed, memcached support not found')

from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, read_from='fs'):
        self.read_from = read_from
        self.initialized = False

    def _init_memcached(self):
        if not self.initialized:
            server_list_config_file = "/..absolute/..path/..to/memcached_client/server_list.conf"  # optional if you use memcached (read_from='mc')
            client_config_file = "/..absolute/..path/..to/memcached_client/client.conf"
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file, client_config_file)
            self.initialized = True

    def read_file(self, filepath):
        if self.read_from == 'mc':
            self._init_memcached()
            value = mc.pyvector()
            self.mclient.Get(filepath, value)
            value_str = mc.ConvertBuffer(value)
            filebytes = np.frombuffer(value_str.tobytes(), dtype=np.uint8)
        elif self.read_from == 'fs':
            filebytes = np.fromfile(filepath, dtype=np.uint8)
        else:
            raise RuntimeError("unknown value for read_from: {}".format(self.read_from))

        return filebytes
