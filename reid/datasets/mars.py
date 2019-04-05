from __future__ import print_function, absolute_import
import os.path as osp
import scipy
from tqdm.autonotebook import tqdm

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class Mars(Dataset):
    fname = [
        'bbox_train.zip',
        'bbox_test.zip'
    ]
    url = [
        'https://drive.google.com/file/d/0B6tjyrV1YrHeY0hsVExLOTk3eVU/view',
        'https://drive.google.com/file/d/0B6tjyrV1YrHeTEE2c2hFMTdpRFU/view'
    ]
    md5 = [
        'f4a3c5967a1b440ccf770f388c609602',
        '950d3bfbd792103d12729ac4f57199cf'
    ]

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(Mars, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        for fname, url, md5 in zip(self.fname, self.url, self.md5):
            # Download the raw zip file
            fpath = osp.join(raw_dir, fname)
            if osp.isfile(fpath): # and \
              # hashlib.md5(open(fpath, 'rb').read()).hexdigest() == md5:
                print("Using downloaded file: " + fpath)
            else:
                raise RuntimeError("Please download the dataset manually from {} "
                                   "to {}".format(url, fpath))

            # Extract the file
            if not osp.isdir(osp.join(raw_dir, 'bbox_train')):
                print("Extracting zip file")
                with ZipFile(fpath) as z:
                    z.extractall(path=raw_dir)

        infodir = osp.join(raw_dir, 'info')
        if not osp.isdir(infodir):
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format('https://github.com/liangzheng06/MARS-evaluation/tree/master/info', infodir))

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 1501 identities (+1 for background) with 6 camera views each
        identities = [[[] for _ in range(6)] for _ in range(1502)]

        tracks_test_info = scipy.io.loadmat(osp.join(infodir, 'tracks_test_info.mat'))['track_test_info']
        query_idx = scipy.io.loadmat(osp.join(infodir, 'query_IDX.mat'))['query_IDX']
        query_idx = set(query_idx[0, :])

        with open(osp.join(infodir, 'test_name.txt'), 'r') as fin:
            test_name = fin.read().split()

        query_fnames = set()

        for tracklet_id, (begin, end, y, cam) in enumerate(tracks_test_info):
            if tracklet_id + 1 in query_idx:
                for fname in test_name[begin - 1:end]:
                    query_fnames.add(fname)

        def register(subdir, pattern=re.compile(r'([-\d]+)C(\d)'), include=lambda x: True):
            fpaths = sorted(glob(osp.join(raw_dir, subdir, '*', '*.jpg')))
            pids = set()
            for fpath in tqdm(fpaths, desc=subdir):
                fname = osp.basename(fpath)
                pid, cam = map(lambda x: int(x) if x != '00-1' else -1, pattern.search(fname).groups())
                if pid == -1: continue  # junk images are just ignored
                assert 0 <= pid <= 1501  # pid == 0 means background
                assert 1 <= cam <= 6
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
            return pids

        trainval_pids = register('bbox_train')
        gallery_pids = register('bbox_test', include=lambda x: x not in query_fnames)
        query_pids = register('bbox_test', include=lambda x: x in query_fnames)
        assert query_pids <= gallery_pids
        assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'Mars', 'shot': 'multiple', 'num_cameras': 6,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
