STABLE_SIMPLE_PICK_ASSET_SPECS = [
    ("047_mouse", [0, 1, 2]),
    ("048_stapler", [0, 1, 2, 3, 4, 5, 6]),
    ("050_bell", [0, 1]),
    ("057_toycar", [0, 1, 2, 3, 4, 5]),
    ("071_can", [0, 1, 2, 3, 5, 6]),
    ("073_rubikscube", [0, 1, 2]),
    ("075_bread", [0, 1, 2, 3, 4, 5, 6]),
    ("077_phone", [0, 1, 2, 3]),
    ("079_remotecontrol", [0, 1, 2, 3, 4, 5, 6]),
    ("080_pillbottle", [1, 2, 3, 4, 5]),
    ("081_playingcards", [0, 1, 2]),
    ("107_soap", [0, 1, 2, 3]),
    ("112_tea-box", [0, 1, 2, 3, 4, 5]),
    ("113_coffee-box", [0, 1, 2, 3, 4, 5, 6]),
]

MIN_UNIQUE_ASSET_EPISODES = 50


def build_interleaved_stable_asset_catalog(get_available_model_ids):
    catalog = []

    for modelname, model_ids in STABLE_SIMPLE_PICK_ASSET_SPECS:
        available_model_ids = set(get_available_model_ids(modelname))
        missing_model_ids = [model_id for model_id in model_ids if model_id not in available_model_ids]
        if missing_model_ids:
            raise RuntimeError(
                f"Stable asset whitelist is missing variants for {modelname}: {missing_model_ids}"
            )

    max_variant_count = max(len(model_ids) for _, model_ids in STABLE_SIMPLE_PICK_ASSET_SPECS)
    for variant_rank in range(max_variant_count):
        for modelname, model_ids in STABLE_SIMPLE_PICK_ASSET_SPECS:
            if variant_rank >= len(model_ids):
                continue
            model_id = model_ids[variant_rank]
            catalog.append(
                {
                    "modelname": modelname,
                    "model_id": model_id,
                    "asset_key": f"{modelname}/base{model_id}",
                }
            )

    if len(catalog) < MIN_UNIQUE_ASSET_EPISODES:
        raise RuntimeError(
            f"Stable asset whitelist only contains {len(catalog)} assets, fewer than "
            f"{MIN_UNIQUE_ASSET_EPISODES}"
        )

    return catalog


def stable_asset_catalog_index(task_name, ep_num, catalog_size):
    if catalog_size < MIN_UNIQUE_ASSET_EPISODES:
        raise RuntimeError(
            f"Asset catalog must contain at least {MIN_UNIQUE_ASSET_EPISODES} assets, got {catalog_size}"
        )

    task_offset = sum(ord(ch) for ch in str(task_name)) % catalog_size
    return (task_offset + int(ep_num)) % catalog_size
