import os
import sys
import h5py
import open3d
import random
import numpy as np
import pandas as pd
from matplotlib import cm


def _resample(arr, num, return_indexes=False):
    replace = False
    if num <= 0:
        num = arr.shape[0]
    if num > arr.shape[0]:
        replace = True
    ixs = np.random.choice(arr.shape[0], num, replace=replace)
    if return_indexes:
        return arr[ixs], ixs
    return arr[ixs]


def _create_colors_list(size, seed, cmap_name='gist_rainbow'):
    cmap = cm.get_cmap(cmap_name, size)
    colors = [cmap(i)[:3] for i in range(size)]
    if seed > -1:
        rstate = random.getstate()
        random.seed(seed)
        random.shuffle(colors)
        random.setstate(rstate)
    return colors



class _PCDataFields:

    FIELD_XYZ = 'xyz'
    FIELD_RGB = 'rgb'
    FIELD_XYZ_NORMALIZED = 'xyz_normalized'
    FIELD_SEMANTIC_CLASS = 'semantic_class'
    FIELD_INSTANCE_ID = 'instance_id'

    @staticmethod
    def get_fields():
        return [_PCDataFields.FIELD_XYZ,
                _PCDataFields.FIELD_RGB,
                _PCDataFields.FIELD_XYZ_NORMALIZED,
                _PCDataFields.FIELD_SEMANTIC_CLASS,
                _PCDataFields.FIELD_INSTANCE_ID]


class PCData:

    FIELDS = _PCDataFields.get_fields()

    FIELD_XYZ = _PCDataFields.FIELD_XYZ
    FIELD_RGB = _PCDataFields.FIELD_RGB
    FIELD_XYZ_NORMALIZED = _PCDataFields.FIELD_XYZ_NORMALIZED
    FIELD_SEMANTIC_CLASS = _PCDataFields.FIELD_SEMANTIC_CLASS
    FIELD_INSTANCE_ID = _PCDataFields.FIELD_INSTANCE_ID

    SYNONIMS = {
        'xyznorm': _PCDataFields.FIELD_XYZ_NORMALIZED,
        'xyz_norm': _PCDataFields.FIELD_XYZ_NORMALIZED,
        'sid': _PCDataFields.FIELD_SEMANTIC_CLASS,
        'gid': _PCDataFields.FIELD_SEMANTIC_CLASS,
        'sem': _PCDataFields.FIELD_SEMANTIC_CLASS,
        'semantic_label': _PCDataFields.FIELD_SEMANTIC_CLASS,
        'semantic_labels': _PCDataFields.FIELD_SEMANTIC_CLASS,
        'semid': _PCDataFields.FIELD_SEMANTIC_CLASS,
        'iid': _PCDataFields.FIELD_INSTANCE_ID,
        'instid': _PCDataFields.FIELD_INSTANCE_ID,
        'pid': _PCDataFields.FIELD_INSTANCE_ID,
        'instance': _PCDataFields.FIELD_INSTANCE_ID,
        'instance_labels': _PCDataFields.FIELD_INSTANCE_ID,
        'instance_label': _PCDataFields.FIELD_INSTANCE_ID,
        'instances': _PCDataFields.FIELD_INSTANCE_ID,
    }

    COLORS_LIST = None

    @staticmethod
    def reset_colors(size=24, seed=4):
        PCData.COLORS_LIST = _create_colors_list(size, seed)

    @staticmethod
    def all_fields():
        return PCData.FIELDS

    @staticmethod
    def valid_fields():
        fields = list(PCData.FIELDS)
        fields.extend(PCData.SYNONIMS.keys())
        return fields

    def __init__(self, data=None):
        self._data = None
        self._path = None
        if data is not None:
            if isinstance(data, str):
                self.load(data)
            elif isinstance(data, np.ndarray):
                self.data = data
            else:
                raise Exception(f'unknown data format: {type(data)}')

    def load(self, path):
        ext = os.path.splitext(path)[1]
        if ext == '.h5':
            self._data = {}
            with h5py.File(path, 'r') as fin:
                for field in self.FIELDS:
                    self._data[field] = fin[field][:]
                self._fix_shape2d()
        elif ext == '.txt':
            self.data = np.loadtxt(path)
        elif ext in ['.ply', '.pcd', '.xyz', '.xyzrgb', '.pts']:
            self.data = PCData._load_pc_data(path)
        elif ext == '.csv':
            df = pd.read_csv(path, header=[0, 1])
            self._data = {}
            fields = set([f[0] for f in df.columns[1:]])
            for field in PCData.FIELDS:
                if field in fields:
                    self._data[field] = df[field].values
            self._fix_shape2d()
        self._path = path

    @staticmethod
    def _load_pc_data(path):
        pc = open3d.io.read_point_cloud(path)
        if pc.has_colors():
            return np.concatenate((np.asarray(pc.points), np.asarray(pc.colors)), axis=1)
        return np.asarray(pc.points)

    def keys(self):
        return self._data.keys()

    def __str__(self):
        desc = f'''fields: {list(self.keys())}
shape: {self.shape}
'''
        if self._path:
            desc += f'source path: {self._path}'
        return desc

    @property
    def shape(self):
        shapes = []
        for arr in self._data.values():
            shapes.append(arr.shape)
        length = shapes[0][0]
        width = 0
        for shape in shapes:
            width += shape[1]
            if length != shape[0]:
                raise Exception("inconsistent shape detected")
        return length, width

    def present(self, fields):
        if isinstance(fields, str):
            fields = [fields]
        present_fields = self._data.keys()
        for field in fields:
            if field not in present_fields:
                return False
        return True

    def values(self):
        return self._data.values()

    @property
    def points(self):
        return self._data[PCData.FIELD_XYZ]

    @property
    def colors(self):
        return self._data[PCData.FIELD_RGB]

    @property
    def semantics(self):
        return self._data[PCData.FIELD_SEMANTIC_CLASS]

    @property
    def instances(self):
        return self._data[PCData.FIELD_INSTANCE_ID]

    @property
    def data(self):
        return np.hstack(list(self.values()))

    @data.setter
    def data(self, arr):
        self._data = {
            PCData.FIELD_XYZ: arr[:, :3]
        }
        if arr.shape[1] == 11:
            self._data[PCData.FIELD_RGB] = arr[:, 3:6]
            self._data[PCData.FIELD_XYZ_NORMALIZED] = arr[:, 6:9]
            self._data[PCData.FIELD_SEMANTIC_CLASS] = arr[:, 9:10]
            self._data[PCData.FIELD_INSTANCE_ID] = arr[:, 10:]
        elif arr.shape[1] == 8:
            self._data[PCData.FIELD_RGB] = arr[:, 3:6]
            self._data[PCData.FIELD_SEMANTIC_CLASS] = arr[:, 6:7]
            self._data[PCData.FIELD_INSTANCE_ID] = arr[:, 7:8]
        elif arr.shape[1] == 6:
            self._data[PCData.FIELD_RGB] = arr[:, 3:6]
        elif arr.shape[1] == 5:
            self._data[PCData.FIELD_SEMANTIC_CLASS] = arr[:, 3:4]
            self._data[PCData.FIELD_INSTANCE_ID] = arr[:, 4:5]
        elif arr.shape[1] == 3:
            pass
        else:
            raise Exception(f'unknown data shape: {arr.shape}')

    @staticmethod
    def create_from_data(xyz, *, rgb=None, xyz_norm=None, seg_label=None, inst_label=None):
        return PCData().set_data(xyz, rgb=rgb, xyz_norm=xyz_norm, seg_label=seg_label, inst_label=inst_label)

    @staticmethod
    def save_data(ofile, xyz, *, rgb=None, xyz_norm=None, seg_label=None, inst_label=None):
        PCData. \
            create_from_data(xyz, rgb=rgb, xyz_norm=xyz_norm, seg_label=seg_label, inst_label=inst_label). \
            save(ofile)

    def set_data(self, xyz, *, rgb=None, xyz_norm=None, seg_label=None, inst_label=None):
        self._data = {
            PCData.FIELD_XYZ: xyz,
        }
        if rgb is not None:
            self._data[PCData.FIELD_RGB] = rgb
        if xyz_norm is not None:
            self._data[PCData.FIELD_XYZ_NORMALIZED] = xyz_norm
        if seg_label is not None:
            self._data[PCData.FIELD_SEMANTIC_CLASS] = seg_label.reshape((-1, 1))
        if inst_label is not None:
            self._data[PCData.FIELD_INSTANCE_ID] = inst_label.reshape((-1, 1))
        return self

    def _fix_shape2d(self):
        for k in self._data:
            if len(self._data[k].shape) == 1:
                self._data[k] = self._data[k].reshape((-1, 1))

    def save(self, path=None, color_channel=_PCDataFields.FIELD_RGB):
        if path is None:
            path = self._path
        if path is None:
            raise Exception('no path to save data')
        if not self.has_data():
            raise Exception('no data to save')

        ext = os.path.splitext(path)[1]
        if ext == '.h5':
            self._save_as_h5(path)
        elif ext == '.txt':
            self._save_as_txt(path)
        elif ext == '.csv':
            self._save_as_csv(path)
        elif ext in ['.xyz', '.pts', '.ply', '.pcd', '.xyzrgb']:
            self.save_as_pc(path, color_channel)
        elif ext in ['.obj']:
            self.save_as_mesh(path, color_channel)

    def _save_as_h5(self, path):
        with h5py.File(path, 'w') as fin:
            for field in self.FIELDS:
                fin.create_dataset(field, data=self._data[field])

    def _save_as_txt(self, path):
        np.savetxt(path, self.data)

    def _save_as_csv(self, path):
        fields = [PCData.FIELD_XYZ, PCData.FIELD_XYZ, PCData.FIELD_XYZ]
        subfields = ['x', 'y', 'z']
        data = self._data[PCData.FIELD_XYZ]
        if PCData.FIELD_RGB in self._data:
            fields.extend([PCData.FIELD_RGB, PCData.FIELD_RGB, PCData.FIELD_RGB])
            subfields.extend(['r', 'g', 'b'])
            data = np.concatenate((data, self._data[PCData.FIELD_RGB]), axis=1)
        if PCData.FIELD_XYZ_NORMALIZED in self._data:
            fields.extend([PCData.FIELD_XYZ_NORMALIZED, PCData.FIELD_XYZ_NORMALIZED, PCData.FIELD_XYZ_NORMALIZED])
            subfields.extend(['x', 'y', 'z'])
            data = np.concatenate((data, self._data[PCData.FIELD_XYZ_NORMALIZED]), axis=1)
        if PCData.FIELD_SEMANTIC_CLASS in self._data:
            fields.append(PCData.FIELD_SEMANTIC_CLASS)
            subfields.append(PCData.FIELD_SEMANTIC_CLASS)
            data = np.concatenate((data, self._data[PCData.FIELD_SEMANTIC_CLASS]), axis=1)
        if PCData.FIELD_INSTANCE_ID in self._data:
            fields.append(PCData.FIELD_INSTANCE_ID)
            subfields.append(PCData.FIELD_INSTANCE_ID)
            data = np.concatenate((data, self._data[PCData.FIELD_INSTANCE_ID]), axis=1)

        titels = list(zip(fields, subfields))
        index = pd.MultiIndex.from_tuples(titels, names=['field', 'subfield'])
        df = pd.DataFrame(data=data, columns=index)
        df.to_csv(path)

    @staticmethod
    def _labels2rgb(labels):
        ulabels, counts = np.unique(labels, return_counts=True)
        ixs = np.argsort(counts)
        ulabels = ulabels[ixs][::-1]
        if PCData.COLORS_LIST is None or ulabels.shape[0] > len(PCData.COLORS_LIST) :
            PCData.reset_colors(ulabels.shape[0] + 3)
        colors = np.empty((labels.shape[0], 3))
        for i, label in enumerate(ulabels):
            colors[(labels == label).reshape(-1)] = PCData.COLORS_LIST[i]
        return colors

    def _get_colors(self, channel=_PCDataFields.FIELD_RGB):
        colors = None
        if channel is None or channel == PCData.FIELD_RGB:
            if self.present(PCData.FIELD_RGB):
                colors = self._data[PCData.FIELD_RGB]
        if channel == PCData.FIELD_SEMANTIC_CLASS and self.present(PCData.FIELD_SEMANTIC_CLASS):
            colors = self._labels2rgb(self._data[PCData.FIELD_SEMANTIC_CLASS])
        if channel == PCData.FIELD_INSTANCE_ID and self.present(PCData.FIELD_INSTANCE_ID):
            colors = self._labels2rgb(self._data[PCData.FIELD_INSTANCE_ID])
        return colors

    def save_as_pc(self, path, color_channel=_PCDataFields.FIELD_RGB):
        pc = open3d.geometry.PointCloud()
        pc.points = open3d.open3d.utility.Vector3dVector(self._data[PCData.FIELD_XYZ])
        colors = self._get_colors(color_channel)
        if colors is not None:
            pc.colors = open3d.open3d.utility.Vector3dVector(colors)
        open3d.io.write_point_cloud(path, pc, write_ascii=True)

    def save_as_mesh(self, path, color_channel=_PCDataFields.FIELD_RGB):
        pc = open3d.geometry.TriangleMesh()
        pc.vertices = open3d.open3d.utility.Vector3dVector(self._data[PCData.FIELD_XYZ])
        colors = self._get_colors(color_channel)
        if colors is not None:
            pc.vertex_colors = open3d.open3d.utility.Vector3dVector(colors)
        open3d.io.write_triangle_mesh(path, pc, write_ascii=True)

    def has_data(self):
        return self._data is not None

    def get_instance_indexes(self, instance_id):
        return self._data[self.FIELD_INSTANCE_ID] == instance_id

    def get_class_indexes(self, class_id):
        return self._data[self.FIELD_SEMANTIC_CLASS] == class_id

    def get(self, field):
        if not self.has_data():
            raise Exception('no data')
        if field in PCData.FIELDS:
            return self._data[field]
        if field in self.SYNONIMS.keys():
            return self._data[self.SYNONIMS[field]]
        raise Exception(f'invalid data-field: {field}')

    def set(self, field, data):
        if field in PCData.FIELDS:
            pass
        elif field in self.SYNONIMS.keys():
            field = self.SYNONIMS[field]
        else:
            raise Exception(f'invalid data-field: {field}')
        if self._data is None:
            self._data = {}
        self._data[field] = data

    def strip_normalized_xyz(self):
        try:
            del self._data[PCData.FIELD_XYZ_NORMALIZED]
        except:
            pass

    def add_jitter(self, maxval):
        jitter = np.random.uniform(0, maxval, size=self._data[PCData.FIELD_XYZ].shape)
        self._data[PCData.FIELD_XYZ] += jitter
        if self.present(PCData.FIELD_XYZ_NORMALIZED):
            self.insert_normalized_xyz(skip_exists=False)

    def insert_normalized_xyz(self, skip_exists=True):
        if skip_exists and PCData.FIELD_XYZ_NORMALIZED in self._data:
            return
        xyz = self._data[PCData.FIELD_XYZ]
        minxyz = np.min(xyz, axis=0)
        maxxyz = np.max(xyz, axis=0)
        self._data[PCData.FIELD_XYZ_NORMALIZED] = (xyz - minxyz) / (maxxyz - minxyz)

    def make_instance_id_zerobased(self):
        labels = self._data[PCData.FIELD_SEMANTIC_CLASS].astype(np.int)
        insts = self._data[PCData.FIELD_INSTANCE_ID].astype(np.int)
        ulabels = np.unique(labels)
        for label in ulabels:
            min_inst = np.min(insts[labels == label])
            insts[labels == label] -= min_inst
        self._data[PCData.FIELD_INSTANCE_ID] = insts

    def make_instance_id_unique(self):
        if self.instance_id_unique():
            return
        labels = self._data[PCData.FIELD_SEMANTIC_CLASS].astype(np.int)
        insts = self._data[PCData.FIELD_INSTANCE_ID].astype(np.int)
        ulabels = np.unique(labels)
        max_inst = 0
        for label in ulabels:
            insts[labels == label] += max_inst
            max_inst = np.max(insts[labels == label]) + 1
        self._data[PCData.FIELD_INSTANCE_ID] = insts

    def instance_id_unique(self):
        if not self.has_data():
            raise Exception('no data yet')
        labels = self._data[PCData.FIELD_SEMANTIC_CLASS].astype(np.int)
        inst = self._data[PCData.FIELD_INSTANCE_ID].astype(np.int)
        ulabels = np.unique(labels)
        return np.min(inst[labels == ulabels[0]]) != np.min(inst[labels == ulabels[1]])

    def __getitem__(self, field):
        return self.get(field)

    def __setitem__(self, field, data):
        self.set(field, data)

    def __len__(self):
        return next(iter(self._data.values())).shape[0]

    def __contains__(self, item):
        return item in self._data

    def resample(self, num_samples):
        ixs = None
        for k in self._data:
            if ixs is None:
                self._data[k], ixs = _resample(self._data[k], num_samples, return_indexes=True)
            else:
                self._data[k] = self._data[k][ixs]

    @staticmethod
    def data2pc(ifile, ofile, num_samples=-1):
        df = PCData()
        df.load(ifile)
        if num_samples > 0:
            points = _resample(df[PCData.FIELD_XYZ], num_samples)
        else:
            points = df[PCData.FIELD_XYZ]
        pc = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
        open3d.io.write_point_cloud(ofile, pc)


if __name__ == '__main__':
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('-i', '--input', required=True,
                            help='input pc-data file path, supported formats: [h5, txt, csv]')
        parser.add_argument('-o', '--output', required=False,
                            help='output pc-data file path, supported formats: [h5, txt, csv]')
        parser.add_argument('-r', '--resample', required=False, type=int,
                            help='shuffle and alter the number of points')
        parser.add_argument('-j', '--jitter', default=-1., type=float,
                            help='add a random shift to the points')
        parser.add_argument('-z', '--make-instance-id-zerobased', action='store_true',
                            help='make instance id labels zero-based')
        parser.add_argument('-u', '--make-instance-id-unique', action='store_true',
                            help='make instance id unique')
        parser.add_argument('-p', '--print-info', action='store_true',
                            help='print point cloud info')
        parser.add_argument('-s', '--strip-normalized-xyz', action='store_true',
                            help='strip normalized xyz')
        parser.add_argument('-a', '--add-normalized-xyz', action='store_true',
                            help='add normalized xyz')
        return parser.parse_args()


    args = get_args()
    pcd = PCData(args.input)
    if args.print_info:
        print(pcd)
    if args.resample is not None and args.resample > 0:
        pcd.resample(args.resample)
    if args.make_instance_id_zerobased and args.make_instance_id_unique:
        print('warning: contradicting options: "make_instance_id_zerobased", "make_instance_id_unique"', file=sys.stderr)
    if args.make_instance_id_zerobased:
        pcd.make_instance_id_zerobased()
    if args.make_instance_id_unique:
        pcd.make_instance_id_unique()
    if args.strip_normalized_xyz and args.add_normalized_xyz:
        print('warning: striping and then adding normalized xyz', file=sys.stderr)
    if args.strip_normalized_xyz:
        pcd.strip_normalized_xyz()
    if args.add_normalized_xyz:
        pcd.insert_normalized_xyz()
    if args.jitter > 0:
        pcd.add_jitter(args.jitter)
    if args.output:
        pcd.save(args.output)

