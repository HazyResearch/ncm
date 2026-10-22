"""Validate shard boundaries before computing any aggregate metric."""

import pickle


def load_shards(paths, *, unconditional=False, run_id=None):
    parts = []
    indices = set()
    seeds = set()
    policy = None
    shard_numbers = set()
    for path in paths:
        with open(path, "rb") as stream:
            part = pickle.load(stream)
        n = len(part["hyps"])
        for key in ("idxs", "refs"):
            if len(part[key]) != n:
                raise ValueError(f"{path}: {key} does not align with hyps")
        for key in ("seeds", "toks", "raw_ids"):
            if key in part and len(part[key]) != n:
                raise ValueError(f"{path}: {key} does not align with hyps")
        if any(type(i) is not int or i < 0 for i in part["idxs"]):
            raise ValueError(f"{path}: indices must be nonnegative integers")
        if len(set(part["idxs"])) != n:
            raise ValueError(f"{path}: duplicate indices within a shard")
        if unconditional:
            if any(key not in part for key in ("seeds", "toks", "raw_ids")):
                raise ValueError(f"{path}: unconditional shard requires seeds, toks and raw_ids")
            current = set(part["seeds"])
            if seeds & current:
                raise ValueError("the same sampling seed occurs in multiple shards")
            seeds.update(current)
        else:
            if indices & set(part["idxs"]):
                raise ValueError("duplicate sample indices across shards")
            indices.update(part["idxs"])

        metadata = part.get("metadata")
        if run_id is not None and (metadata is None or metadata.get("run_id") != run_id):
            raise ValueError(f"{path}: shard belongs to a different run")
        current_policy = None if metadata is None else {
            key: value for key, value in metadata.items() if key != "shard"
        }
        if parts and current_policy != policy:
            raise ValueError("shards have inconsistent configuration, policy or provenance")
        policy = current_policy
        if metadata is not None:
            number = metadata["shard"]
            if number in shard_numbers:
                raise ValueError("duplicate shard number")
            shard_numbers.add(number)
        parts.append(part)

    if not parts or not any(part["hyps"] for part in parts):
        raise ValueError("no samples in shard files")
    expected_inputs = policy.get("n_inputs", len(indices)) if policy is not None else len(indices)
    if not unconditional and indices != set(range(expected_inputs)):
        raise ValueError("missing conditional sample indices")
    if policy is not None and shard_numbers != set(range(policy["nshards"])):
        raise ValueError("missing shard files")
    return parts
