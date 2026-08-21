from torch import nn


class ModelCollator:

    def __init__(self, source_name: str, source_model : nn.Module, variants: list[nn.Module]):
        self.source_model = source_model
        self.source_name = source_name
        self.layers = self.set_source_layers()
        for ind, variant in enumerate(variants):
            self.verify_variant(ind, variant)
        self.variants = variants

    def set_source_layers(self):
        layers = {layer: weight.shape for layer, weight in self.source_model.state_dict().items()}
        return layers

    def verify_variant(self, var_name, variant : nn.Module):
        if not isinstance(variant, type(self.source_model)):
            raise TypeError(f"Variant {var_name} got {type(variant)}, expecting {type(self.source_model)}")

        layer = {lay_name : weight.shape for lay_name, weight in variant.state_dict().items()}

        for src_layer, var_layer in zip(self.layers, layer):
            src_shape = self.layers[src_layer]
            var_shape = layer[var_layer]
            if src_shape != var_shape:
                raise ValueError(f"Expecting layer {src_layer} with shape {src_shape}, got {var_layer} with shape {var_shape} on Variant {var_name}")
            if src_layer != var_layer:
                print(f"Variant {var_name} has different layer name {var_layer} with source layer {src_layer}, however it has the same shape")

    def __iter__(self):
        yield from self.layers

    def __getitem__(self, key):
        return self.variants[key]

    def get_layer(self, ind, layer):
        variant =  self.variants[ind]
        state = variant.state_dict()
        return state[layer]
