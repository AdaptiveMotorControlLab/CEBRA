#
# Regularized contrastive learning implementation.
#
# Not licensed yet. Distribution for review.
# Code will be open-sourced upon publication.
#
import dataclasses
from typing import Dict, List, Literal, Tuple

import sklearn
import torch
from sklearn.metrics import r2_score

from cebra.solver import init
from cebra.solver import register


@dataclasses.dataclass
class MetricCollection():
    metrics: List["Metric"]
    datasets: Dict[Literal["train", "val"], "DatasetxCEBRA"]
    #NOTE: we need datasets for _compute_metrics()

    splits: Tuple[str] = ("train", "val")

    @classmethod
    def from_config(cls, config, datasets, splits=("train", "val")):

        metrics = []
        for metric_config in config:
            kwargs = metric_config['kwargs']

            labels = {}
            for split in splits:
                labels[split] = datasets[split].labels[
                    kwargs["label"]].detach().cpu().numpy()

            metric = init(name=metric_config['metric_name'],
                          labels=labels,
                          label_name=kwargs['label'],
                          indices=kwargs['indices'])
            metrics.append(metric)

        return MetricCollection(metrics=metrics, datasets=datasets)

    def compute_metrics(self, embeddings):
        result = {}
        for metric in self.metrics:
            #NOTE: model is always based on train data.
            metric.fit(embeddings["train"])
            for split in self.splits:
                metric_result = metric.score(embeddings[split], split)
                result[f"{metric.name}_{split}", metric.label_name,
                       metric.indices] = metric_result
        return result


#
@dataclasses.dataclass
class Metric():
    labels: Dict[Literal["train", "val"], torch.Tensor]
    label_name: str
    indices: Tuple

    def fit(self, embedding: torch.Tensor, split: Literal["train", "val"]):
        raise NotImplementedError()

    def score(self, embedding: torch.Tensor, split: Literal["train",
                                                            "val"]) -> float:
        raise NotImplementedError()


@dataclasses.dataclass
@register("r2_linear")
class LinearRegressionScore(Metric):

    def __post_init__(self):
        self.name = "r2"
        self._model = sklearn.linear_model.LinearRegression()

    def fit(self, embedding, split="train"):
        self._model.fit(embedding[:, slice(*self.indices)], self.labels[split])

    def score(self, embedding: torch.Tensor, split: Literal["train",
                                                            "val"]) -> float:
        prediction = self._model.predict(embedding[:, slice(*self.indices)])
        score = r2_score(self.labels[split], prediction)
        return score
