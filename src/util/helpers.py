def create_json_dict(d: dict) -> dict:
    return {k: v for k, v in d.items() if isinstance(k, str) and safe_json(v)}


def safe_json(data) -> bool:
    if data is None:
        return True
    elif isinstance(data, (bool, int, float)):
        return True
    elif isinstance(data, (tuple, list)):
        return all(safe_json(x) for x in data)
    elif isinstance(data, dict):
        return all(isinstance(k, str) and safe_json(v) for k, v in data.items())
    return False
