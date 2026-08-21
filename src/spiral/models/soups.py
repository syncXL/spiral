import copy
from typing import Literal

import torch
from torch import nn
from torch.func import functional_call

from spiral.utils.inference import Evaluate
from spiral.utils.model import ModelCollator


class ModelSoup:
    def __init__(self,
        models : ModelCollator,
        eval: Evaluate,
        strategy: Literal["uniform", "greedy"],
        order: Literal["desc","random"] | None = None,
        metrics: list[float] | None = None):

        self.models = models
        self.eval = eval
        self.strategy = strategy
        self.order = order
        self.metrics = []
        self.souped_model = None

        if strategy == "greedy" and not order:
            self.order = "desc"

        if metrics:
            self.metrics = torch.tensor(metrics)
            if len(metrics) != len(models.variants):
                raise ValueError(f"Got {len(models.variants)} and {len(metrics)}")

    def order_metrics(self, k):
        if len(self.metrics) == 0:
            return []
        if self.strategy == "greedy" and self.order == "desc":
            ordered_metrics_args = self.metrics.argsort(descending=True).tolist()
            return ordered_metrics_args[:k]
        return list(range(self.metrics.shape[0]))[:k]

    def compute_metrics(self, data):
        metrics = []
        for model in self.models.variants:
            score = self.eval(model, data)
            metrics.append(score)
        self.metrics = torch.tensor(metrics)

    def update_weight_model(self, state_model, model_id, ind):
        for layer in self.models:
            init_weight = state_model[layer]
            cur_weight = self.models.get_layer(model_id, layer)
            k = (ind + 2)
            momentum_k = (k - 1) / k
            state_model[layer] = (momentum_k * init_weight) + ((1 / k) * cur_weight)
        return state_model

    def soupify(self, args_ordered, data):
        # loop layer
        # get weight for each model
        self.souped_model = copy.deepcopy(self.models.variants[args_ordered[0]])
        soup_state = copy.deepcopy(self.souped_model.state_dict())
        for ind,model_id in enumerate(args_ordered[1:]):
            soup_state = copy.deepcopy(self.souped_model.state_dict())
            upd_state = self.update_weight_model(copy.deepcopy(soup_state), model_id, ind)
            prop_model = copy.deepcopy(self.souped_model)
            prop_model.load_state_dict(upd_state)
            if self.strategy == "greedy":
                soup_metric = self.eval(self.souped_model, data)
                prop_model_metric = self.eval(prop_model, data)
                if prop_model_metric >= soup_metric:
                    self.souped_model = prop_model
                    continue
            else:
                self.souped_model = prop_model

        self.souped_model.load_state_dict(soup_state)

    def build(self, data = None, k=3):
        if data is not None:
            self.compute_metrics(data)
        if len(self.metrics) == 0:
            raise ValueError("No Metrics or data added")

        
        ordered_metrics = self.order_metrics(k)
        self.soupify(ordered_metrics,data)
        return self.souped_model, ordered_metrics, self.metrics


class LearnedModule(nn.Module):
    def __init__(self, model : ModelCollator, pred_fn: callable,mode : Literal["standard", "per_layer"] = "standard"):
        super().__init__()
        self.variants = model
        self.souped = copy.deepcopy(model.variants[0])
        self.mode = mode
        self.alpha, self.beta = self.init_params()
        self.pred_fn =pred_fn
        self.souped.eval()

    def init_params(self):
        n_models = len(self.variants.variants)
        if self.mode == "standard":
            alphas = self.init_weights(n_models)
        else:
            alphas = nn.ParameterList([self.init_weights(n_models) for l in self.variants])
        beta = nn.Parameter(torch.randn([1,1]))
        return alphas, beta


    def init_weights(self, n_models):
        alpha = nn.Parameter(torch.full((1,n_models),1 / n_models))
        return alpha

    def load_layer(self, layer : str, weight: torch.Tensor):
        weight = weight.squeeze(0)
        n_models = len(self.variants.variants)
        layer_weights = torch.stack([self.variants.get_layer(i, layer) for i in range(n_models)])
        return torch.einsum('m,m...->...',weight, layer_weights)

    def load_weights(self):
        upd_state_dict = {}
        if self.mode == "standard":
            for layer in self.variants:
                upd_state_dict[layer] = self.load_layer(layer, self.alpha)
        elif self.mode == "per_layer":
            for layer, alpha in zip(self.variants, self.alpha):
                upd_state_dict[layer] = self.load_layer(layer, alpha)
        return upd_state_dict

    def forward(self, X):
        soup_state = self.load_weights()
        logits = functional_call(self.souped, soup_state, (X,))
        return logits * self.beta