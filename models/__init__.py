

AVAILABLE_MODELS = ["ast", "yamnet"]
def load_model(name):
    name = name.lower()
    if name == "ast":
        from .ast_model import ASTModel
        return ASTModel()
    elif name == "yamnet":
        from .yamnet_model import YAMNetModel
        return YAMNetModel()
    else:
        raise ValueError(f"Unknown model '{name}'. Choices: {AVAILABLE_MODELS}")
