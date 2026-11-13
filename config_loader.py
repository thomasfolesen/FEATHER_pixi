def load_config(config_path="config.txt"):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            config[key.strip()] = value.strip()
    config["theta_max"]   = float(config["theta_max"])
    config["eval_points"] = int(config["eval_points"])
    config["order"]       = int(config["order"])
    return config

def load_amenities(amenities_path="amenities.txt"):
    amenities = []
    with open(amenities_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                amenities.append(line)
    return amenities