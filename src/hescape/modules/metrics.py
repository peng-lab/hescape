from __future__ import annotations

from typing import Any

import numpy as np
import torch
from anndata import AnnData
from scib_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation
from sklearn.metrics import r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from hescape.constants import DatasetEnum


def knn_recall(
    left_embedding: torch.Tensor,
    right_embedding: torch.Tensor,
    embedding_type: str,
    stage: str,
    recall_range: tuple[int] = (1, 5, 10),
):
    metrics = {}
    group_idx = torch.arange(left_embedding.shape[0], device=left_embedding.device)
    emb_sim = left_embedding @ right_embedding.T
    left_gid = group_idx.unsqueeze(1).expand_as(emb_sim)
    right_gid = group_idx.unsqueeze(0).expand_as(emb_sim)
    right_simrank = torch.argsort(emb_sim, dim=1, descending=True)
    rightgid_sorted = torch.gather(right_gid, 1, right_simrank)  # (N,D)
    rightgid_matched = rightgid_sorted == left_gid

    leftgid_hasmatch, leftgid_firstmatch = torch.max(rightgid_matched, dim=1)
    leftmatch_rank = leftgid_firstmatch[leftgid_hasmatch]
    assert leftmatch_rank.shape[0] > 0
    for recall_bound in recall_range:
        match_count = (leftmatch_rank < recall_bound).sum()
        totcal_count = leftgid_hasmatch.sum()
        metrics[f"{stage}_R@{recall_bound}_{embedding_type}"] = (match_count / totcal_count).item()
    return metrics


class EvalMetrics:
    def __init__(
        self,
        stage: str,
        strategy: str,
        label_key: str | None = None,
        batch_key: str | None = None,
        eval_labels: list[str] | None = None,
        recall_range: tuple[int] = (1, 5, 10),
        knn_recall_metrics: bool = False,
        knn_gexp_metrics: bool = False,
        classif_metrics: bool = False,
        scib_metrics: bool = False,
    ):
        self.label_key = label_key
        self.batch_key = batch_key
        self.stage = stage
        self.eval_labels = eval_labels
        self.recall_range = recall_range
        self.strategy = strategy

        # metrics
        self.knn_recall_metrics = knn_recall_metrics
        self.knn_gexp_metrics = knn_gexp_metrics
        self.classif_metrics = classif_metrics
        self.scib_metrics = scib_metrics

    def knn_recall(
        self,
        left_embedding: torch.Tensor,
        right_embedding: torch.Tensor,
        embedding_type: str,
    ):
        metrics = {}
        group_idx = torch.arange(left_embedding.shape[0], device=left_embedding.device)
        emb_sim = left_embedding @ right_embedding.T
        left_gid = group_idx.unsqueeze(1).expand_as(emb_sim)
        right_gid = group_idx.unsqueeze(0).expand_as(emb_sim)
        right_simrank = torch.argsort(emb_sim, dim=1, descending=True)
        rightgid_sorted = torch.gather(right_gid, 1, right_simrank)  # (N,D)
        rightgid_matched = rightgid_sorted == left_gid

        leftgid_hasmatch, leftgid_firstmatch = torch.max(rightgid_matched, dim=1)
        leftmatch_rank = leftgid_firstmatch[leftgid_hasmatch]
        assert leftmatch_rank.shape[0] > 0
        for recall_bound in self.recall_range:
            match_count = (leftmatch_rank < recall_bound).sum()
            totcal_count = leftgid_hasmatch.sum()
            metrics[f"{self.stage}_R@{recall_bound}_{embedding_type}"] = (match_count / totcal_count).item()
        return metrics

    def knn_gexp(
        self,
        left_embedding: torch.Tensor,  # (N, D)
        right_embedding: torch.Tensor,  # (N, D)
        true_gexp: torch.Tensor,  # (N, G)
        stratify: torch.Tensor,  # (N,)
        groups: torch.Tensor,  # (N,)
        embedding_type: str,  # I2G or G2I
    ):
        metrics = {}
        for k in self.recall_range:  # (1,5,10)
            r2_mean_l = []
            r2_var_l = []
            for group in torch.unique(groups):
                try:  # The least populated class in y has only 1 member
                    if stratify[groups == group].shape[0] > k * 2:
                        (
                            left_emb_train,
                            _,
                            _,
                            right_emb_test,
                            true_gexp_train,
                            true_gexp_test,
                            stratify_train,
                            stratify_test,
                        ) = train_test_split(
                            left_embedding[groups == group],
                            right_embedding[groups == group],
                            true_gexp[groups == group],
                            stratify[groups == group],
                            test_size=0.5,
                            stratify=stratify[groups == group],
                        )
                        emb_sim = left_emb_train @ right_emb_test.T
                        pred_gexp_test = torch.zeros(true_gexp_test.shape)
                        _, topk_idx = torch.topk(emb_sim, k=k, dim=1)
                        # topk_idx = topk_idx.T
                        for i in range(topk_idx.shape[0]):
                            pred_gexp_test[i] = true_gexp_test[topk_idx[i]].mean(0)
                        for cluster in torch.unique(stratify):
                            r2mean, r2var = r2_eval(
                                true_gexp_train[stratify_train == cluster],
                                pred_gexp_test[stratify_test == cluster],
                            )
                        # r2mean, r2var = r2_eval(
                        #     true_gexp_test,
                        #     pred_gexp_test,
                        # )
                        r2_mean_l.append(r2mean)
                        r2_var_l.append(r2var)
                except ValueError:
                    continue
            #     r2mean, r2val = r2_eval(true_gexp_test, pred_gexp_test)
            #     r2_mean_l.append(r2mean)
            #     r2_var_l.append(r2val)

            # for i in range(3):
            #     left_emb_train, _, _, right_emb_test, _, true_gexp_test = train_test_split(
            #         left_embedding, right_embedding, true_gexp, test_size=0.5, stratify=stratify
            #     )
            #     emb_sim = left_emb_train @ right_emb_test.T
            #     pred_gexp_test = torch.zeros(true_gexp_test.shape)
            #     _, topk_idx = torch.topk(emb_sim, k=k, dim=0)
            #     topk_idx = topk_idx.T
            #     for i in range(topk_idx.shape[0]):
            #         pred_gexp_test[i] = true_gexp[topk_idx[i]].mean(0)
            #     r2mean, r2val = r2_eval(true_gexp_test, pred_gexp_test)
            #     r2_mean_l.append(r2mean)
            #     r2_var_l.append(r2val)
            metrics[f"{self.stage}_r2mean@{k}_{embedding_type}"] = np.mean(r2_mean_l)
            metrics[f"{self.stage}_r2var@{k}_{embedding_type}"] = np.mean(r2_var_l)
        return metrics

    def svc_eval(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label: str,
        embedding_type: str,
    ) -> float:
        metrics = {}
        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("knn", SVC(kernel="linear")),
            ]
        )
        if len(np.unique(y)) == 1:
            metrics[f"{self.stage}_acc_{label}_{embedding_type}"] = np.NAN
            return metrics
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        metrics[f"{self.stage}_acc_{label}_{embedding_type}"] = cross_val_score(X=X, y=y, cv=skf, estimator=clf).mean()
        return metrics

    def scib_eval(
        self,
        X: np.ndarray,
        batch_values: np.ndarray,
        label_values: np.ndarray,
        embedding_type: str,
    ) -> float:
        metrics = {}
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        adata = AnnData(
            X,
            obs={
                self.batch_key: batch_values.flatten(),
                self.label_key: label_values.flatten(),
            },
        )
        adata.obsm["pre"] = X.copy()
        adata.obsm[embedding_type] = X.copy()
        bm = Benchmarker(
            adata,
            batch_key=self.batch_key,
            label_key=self.label_key,
            embedding_obsm_keys=[embedding_type],
            bio_conservation_metrics=BioConservation(
                clisi_knn=False,
                isolated_labels=False,
                nmi_ari_cluster_labels_kmeans=True,
                nmi_ari_cluster_labels_leiden=False,
                silhouette_label=True,
            ),
            batch_correction_metrics=BatchCorrection(
                silhouette_batch=True,
                kbet_per_label=False,
                graph_connectivity=True,
                ilisi_knn=True,
                pcr_comparison=False,
            ),
            pre_integrated_embedding_obsm_key="pre",
            n_jobs=10,
        )
        bm.benchmark()
        bm_d = bm.get_results(min_max_scale=False).to_dict()
        metrics.update({f"{self.stage}_{k}_{bm_d[k]['Metric Type']}": bm_d[k][embedding_type] for k in bm_d})
        return metrics

    def __call__(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        metrics = {}

        img_emb = batch[DatasetEnum.IMG_EMBED]
        gexp_emb = batch[DatasetEnum.GEXP_EMBED]

        if self.knn_recall_metrics:
            knn_recall_i2g = self.knn_recall(
                img_emb,
                gexp_emb,
                "I2G",
            )
            knn_recall_g2i = self.knn_recall(
                gexp_emb,
                img_emb,
                "G2I",
            )
            metrics.update(knn_recall_i2g)
            metrics.update(knn_recall_g2i)
        if self.knn_gexp_metrics:
            true_gexp = batch[DatasetEnum.GEXP]
            stratify = batch[DatasetEnum.CLUSTER]
            groups = batch[DatasetEnum.REGION]
            if self.strategy == "ddp":
                stratify = stratify.squeeze(1)
                groups = groups.squeeze(1)
            knn_gexp_i2g = self.knn_gexp(
                left_embedding=img_emb,
                right_embedding=gexp_emb,
                true_gexp=true_gexp,
                stratify=stratify,
                groups=groups,
                embedding_type="I2G",
            )
            knn_gexp_g2i = self.knn_gexp(
                left_embedding=img_emb,
                right_embedding=gexp_emb,
                true_gexp=true_gexp,
                stratify=stratify,
                groups=groups,
                embedding_type="G2I",
            )
            metrics.update(knn_gexp_i2g)
            metrics.update(knn_gexp_g2i)
        if self.classif_metrics and self.eval_labels is not None:
            for k in self.eval_labels:
                label_values = batch[k]
                if self.strategy == "ddp":
                    label_values = label_values.squeeze(1)
                svc_eval_img_metrics = self.svc_eval(
                    img_emb,
                    label_values,
                    k,
                    DatasetEnum.IMG,
                )
                svc_eval_gexp_metrics = self.svc_eval(
                    gexp_emb,
                    label_values,
                    k,
                    DatasetEnum.GEXP,
                )
                metrics.update(svc_eval_img_metrics)
                metrics.update(svc_eval_gexp_metrics)
        if self.scib_metrics:
            label_values = batch[self.label_key]
            batch_values = batch[self.batch_key]
            if self.strategy == "ddp":
                label_values = label_values.squeeze(1)
                batch_values = batch_values.squeeze(1)
            scib_eval_img_metrics = self.scib_eval(
                img_emb,
                batch_values=batch_values,
                label_values=label_values,
                embedding_type=DatasetEnum.IMG,
            )
            scib_eval_gexp_metrics = self.scib_eval(
                gexp_emb,
                batch_values=batch_values,
                label_values=label_values,
                embedding_type=DatasetEnum.GEXP,
            )
            metrics.update(scib_eval_img_metrics)
            metrics.update(scib_eval_gexp_metrics)

        return metrics


def r2_eval(
    true: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    true_mean, pred_mean = np.nanmean(true, axis=0), np.nanmean(predicted, axis=0)
    true_var, pred_var = np.nanvar(true, axis=0), np.nanvar(predicted, axis=0)
    return r2_score(true_mean, pred_mean), r2_score(true_var, pred_var)
