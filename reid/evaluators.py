from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
from tqdm.autonotebook import tqdm

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter


def extract_features(model, data_loader, normalize_features=False):
    model.eval()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(tqdm(data_loader, desc='Extract Features')):
        outputs = extract_cnn_feature(model, imgs, modules=[model.features], normalize_features=normalize_features)[0]
        for fname, output, pid in zip(fnames, outputs, pids):
            features[fname] = output
            labels[fname] = pid

    return features, labels

BATCH_SIZE = 256

class PairwiseDistance:
    def __init__(self, features, query=None, gallery=None, metric=None):
        self.features = features
        self.query = query
        self.gallery = gallery
        self.metric = metric

        if query is None and gallery is None:
            self.shape = (len(features), ) * 2
        else:
            self.shape = (len(query), len(gallery))

    def __iter__(self):
        features = self.features
        query = self.query
        gallery = self.gallery
        metric = self.metric

        if query is None and gallery is None:
            def iterator():
                n = len(features)
                x = torch.cat(list(features.values()))
                x = x.view(n, -1)
                if metric is not None:
                    x = metric.transform(x)
                dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2

                for dist_batch_start in tqdm(range(0, n, BATCH_SIZE), desc='Calc distance'):
                    dist_batch = dist[dist_batch_start:dist_batch_start + BATCH_SIZE]
                    x_batch = x[dist_batch_start:dist_batch_start + BATCH_SIZE]
                    batch_size = dist_batch.shape[0]
                    dist_batch = (dist_batch.expand(batch_size, n)
                                  - 2 * torch.mm(x_batch, x.t())
                                 )

                    for row in dist_batch.cpu().numpy():
                        yield row
        else:
            def iterator():
                x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
                y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
                m, n = x.size(0), y.size(0)
                x = x.view(m, -1)
                y = y.view(n, -1)
                if metric is not None:
                    x = metric.transform(x)
                    y = metric.transform(y)
                x2 = torch.pow(x, 2).sum(dim=1, keepdim=True) # .expand(m, n) + \
                y2 = torch.pow(y, 2).sum(dim=1, keepdim=True) # .expand(n, m).t()

                for dist_batch_start in tqdm(range(0, m, BATCH_SIZE), desc='Calc distance'):
                    x_batch = x[dist_batch_start:dist_batch_start + BATCH_SIZE]
                    x2_batch = x2[dist_batch_start:dist_batch_start + BATCH_SIZE]
                    batch_size = x_batch.shape[0]

                    dist = x2_batch.expand(batch_size, n) + y2.expand(n, batch_size).t()
                    dist.addmm_(1, -2, x_batch, y.t())


                    for row in dist.cpu().numpy():
                        yield row

        return iterator()


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), only_top1=False):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query]
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    if only_top1:
        return cmc(distmat, query_ids, gallery_ids,
                   query_cams, gallery_cams,
                   separate_camera_set=False,
                   single_gallery_shot=False,
                   first_match_break=False)[0]

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    cmc_configs = {
        'allshots': dict(separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False),
        'cuhk03': dict(separate_camera_set=True,
                       single_gallery_shot=True,
                       first_match_break=False),
        'market1501': dict(separate_camera_set=False,
                           single_gallery_shot=False,
                           first_match_break=True)}
    cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                            query_cams, gallery_cams, **params)
                  for name, params in cmc_configs.items()}

    print('CMC Scores{:>12}{:>12}{:>12}'
          .format('allshots', 'cuhk03', 'market1501'))
    for k in cmc_topk:
        print('  top-{:<4}{:12.1%}{:12.1%}{:12.1%}'
              .format(k, cmc_scores['allshots'][k - 1],
                      cmc_scores['cuhk03'][k - 1],
                      cmc_scores['market1501'][k - 1]))

    # Use the allshots cmc top-1 score for validation criterion
    return cmc_scores['allshots'][0]


class Evaluator(object):
    def __init__(self, model, normalize_features=False, only_top1=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.normalize_features = normalize_features
        self.only_top1 = only_top1

    def evaluate(self, data_loader, query, gallery, metric=None):
        features, _ = extract_features(self.model, data_loader, normalize_features=self.normalize_features)
        distmat = PairwiseDistance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery, only_top1=self.only_top1)
