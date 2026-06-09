def build_method_instance(dataset_name, method_name, data_conf, method_conf, seed, data_root, savedir):
    from registry import dataset_registry, method_registry 
    dataset_class = dataset_registry[dataset_name]
    dataset = dataset_class(data_root)
    load_outputs = dataset.load(batch_size=data_conf["batch_size"])

    train_loader0, val_loader0, test_loader0, train_loader1, val_loader1, test_loader1 = load_outputs

    cfg = {**data_conf, **method_conf}
    context = {
        **cfg,
        "train_loader0": train_loader0,
        "train_loader1": train_loader1,
        "val_loader0": val_loader0,
        "val_loader1": val_loader1,
        "test_loader0": test_loader0,
        "test_loader1": test_loader1,
        "seed": seed,
    }

    method_class = method_registry[method_name]
    return method_class(dataset_name, savedir, **context), context
