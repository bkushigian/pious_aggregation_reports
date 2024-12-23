def version() -> str:
    import importlib.metadata

    return importlib.metadata.version("pious_pro")


def pious_version() -> str:
    import importlib.metadata

    return importlib.metadata.version("pious")
