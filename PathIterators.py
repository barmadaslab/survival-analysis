import os
from os.path import isdir, basename, dirname
import re
from glob import glob, iglob
from itertools import groupby

# Mappings from the Path to X.
PATH_TO_NAME = lambda p: basename(p).split('.')[0].split('_')[0]
PATH_TO_TILE = lambda p: int(basename(p).split('.')[0].split('_')[1])
tpnum_pattern = re.compile('T[0-9]+')
PATH_TO_TPNUM = lambda p: int(tpnum_pattern.search(p).group(0)[1:])

dirname_n = lambda n, path: dirname(path) if n == 1 else dirname_n(n-1, dirname(path))

def sorted_groupby(iterable, groupByKey, sortByKey=None):
    return groupby(
            sorted(iterable, key=groupByKey if sortByKey is None else sortByKey), key=groupByKey)

class PathIterator():
    def __init__(self, workdir):
        self.workdir = workdir

class RawIterator(PathIterator):
    path_to_ch = classmethod(lambda _, p: basename(dirname_n(2, p)))
    path_to_unitnum = classmethod(lambda _, p: int(basename(p).split('.')[0].split('_')[1]))
    def tp_by_name_by_channel_by_units(self):
        paths = sorted(iglob(f'{self.workdir}/raw_imgs/*/T*/col_*/*tif'), key=PATH_TO_TPNUM)
        path_to_ch = lambda p: basename(dirname_n(3, p))
        path_to_unitnum = lambda p: int(basename(p).split('.')[0].split('_')[1])
        for tp, tp_paths in sorted_groupby(paths, groupByKey=PATH_TO_TPNUM):
            for name, name_paths in sorted_groupby(tp_paths, groupByKey=PATH_TO_NAME):
                for channel, ch_paths in sorted_groupby(name_paths, groupByKey=path_to_ch):
                    yield tp, name, channel, sorted(ch_paths, key=RawIterator.path_to_unitnum)

    def tp_by_name_by_channel_by_unit_by_frames(self):
        path_to_unitnum = lambda p: int(basename(p).split('.')[0].split('_')[1])
        path_to_frame_num = lambda p: int(basename(p).split('.')[0].split('_')[-1])
        for tp, channel, name, unit_paths in self.tp_by_name_by_channel_by_units():
            for unit, frame_paths in groupby(unit_paths, groupByKey=path_to_unitnum):
                yield tp, channel, name, unit, sorted(frame_paths, key=path_to_frame_num)

class StitchedIterator(PathIterator):
    path_to_ch = classmethod(lambda _, p: basename(dirname_n(2, p)))
    def tp_by_channel(self, primaryChannel=None):
        paths = iglob(f'{self.workdir}/processed_imgs/stitched/*/T*/*tif')
        for tp, tp_paths in sorted_groupby(paths, groupByKey=PATH_TO_TPNUM):
            for channel, ch_paths in sorted_groupby(tp_paths, groupByKey=StitchedIterator.path_to_ch):
                yield tp, channel, ch_paths

    def name_by_channel_by_tps(self, primaryChannel=None):
        # If a primary channel is given, ensure that it is the first channel iterated through.
        if primaryChannel is not None:
            primFirst = lambda p: 'a' if StitchedIterator.path_to_ch(p).lower() == primaryChannel.lower() else 'z'
        else:
            primFirst = StitchedIterator.path_to_ch
        #primFirst = lambda path: 10 if StitchedIterator.path_to_ch == PRIM else 3
        paths = iglob(f'{self.workdir}/processed_imgs/stitched/*/T*/*tif')
        for name, name_paths in sorted_groupby(paths, groupByKey=PATH_TO_NAME):
            for ch, ch_paths in sorted_groupby(name_paths, 
                    groupByKey=StitchedIterator.path_to_ch, sortByKey=primFirst):
                yield name, ch, sorted(ch_paths, key=PATH_TO_TPNUM)

# Provide another set of booleans for filtering.
# improve name
class StackedIterator(PathIterator):
    def name_by_channel(self, channels=[]):
        path_to_ch = lambda p: basename(dirname(p))
        paths = iglob(f'{self.workdir}/processed_imgs/stacked/*/*tif')
        # Filter channels if applicable.
        if channels:
            paths = filter(lambda p: path_to_ch(p) in channels, paths)

        for name, name_paths in sorted_groupby(paths, groupByKey=PATH_TO_NAME):
            for channel, name_path in sorted_groupby(name_paths, groupByKey=path_to_ch):
                yield name, channel, list(name_path)[0]

class TStackedIterator(PathIterator):
    def name_by_tile_by_channel_by_tps(self):
        paths = iglob(f'{self.workdir}/processed_imgs/t-stacked/*/T*/*tif')
        for name, name_paths in sorted_groupby(paths, groupByKey=PATH_TO_NAME):
            for tile, tile_paths in sorted_groupby(name_paths, groupByKey=PATH_TO_TILE):
                for ch, ch_paths in sorted_groupby(tile_paths, groupByKey=StitchedIterator.path_to_ch):
                    yield name, tile, ch, sorted(ch_paths, key=PATH_TO_TPNUM)

    def name_by_channel_by_tps(self):
        paths = iglob(f'{self.workdir}/processed_imgs/t-stacked/*/T*/*tif')
        for name, name_paths in sorted_groupby(paths, groupByKey=PATH_TO_NAME):
            for ch, ch_paths in sorted_groupby(name_paths, groupByKey=StitchedIterator.path_to_ch):
                yield name, ch, sorted(ch_paths, key=PATH_TO_TPNUM)

class ImageIterator():
    path_iterator = { 
        'raw': RawIterator,
        'stacked': StackedIterator,
        'stitched': StitchedIterator,
        't-stacked': TStackedIterator,
    }

    @classmethod
    def rooted_at(cls, workdir, root):
        _cls = cls.path_iterator[root]
        return _cls(workdir)

if __name__ == '__main__':
    # Currently used for running some simple tests.
    workdir = r'j:/bf/bf245_d/'
    iter = ImageIterator.rooted_at(workdir, 'stitched').name_by_channel_by_tps(primaryChannel='gfp')
    for name, ch, tp_paths in iter:
        print(tp_paths)
