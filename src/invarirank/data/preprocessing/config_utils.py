def cfg_get(cfg, path: str, default=None):
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif hasattr(cur, key):
            cur = getattr(cur, key)
        else:
            try:
                cur = cur[key]
            except Exception:
                return default
    return default if cur is None else cur
