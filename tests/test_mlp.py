from with_features.mlp import MLPGenreClassifier

def test():
    model = MLPGenreClassifier()
    assert  isinstance(model, MLPGenreClassifier) 