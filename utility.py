import json

# ===================================================================
# Utility functions for reading config files
# ===================================================================


class DictionaryUtility:
    """
    Utility methods for dealing with dictionaries.
    """
    @staticmethod
    def to_object(item):
        """
        Convert a dictionary to an object (recursive).
        """
        def convert(item):
            if isinstance(item, dict):
                return type('jo', (), {k: convert(v) for k, v in item.items()})
            if isinstance(item, list):
                def yield_convert(item):
                    for index, value in enumerate(item):
                        yield convert(value)
                return list(yield_convert(item))
            else:
                return item

        return convert(item)

    def to_dict(obj):
        """
         Convert an object to a dictionary (recursive).
         """
        def convert(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(convert(item))
                else:
                    element = convert(val)
                result[key] = element
            return result

        return convert(obj)


def get_config(config_path):
    with open(config_path) as json_data_file:
        config = json.load(json_data_file)
    # convert dict to object recursively for easy call
    config = DictionaryUtility.to_object(config)
    if 'LQMFG' in config_path:
        config.eqn_config.dim_x = config.eqn_config.n_player
        config.eqn_config.dim_w = config.eqn_config.n_player + 1
        config.eqn_config.dim_z = config.eqn_config.n_player + 1
    if 'RSG' in config_path:
        config.eqn_config.dim_w = config.eqn_config.dim_x
        config.eqn_config.dim_z = config.eqn_config.dim_x
    if 'Covid' in config_path:
        config.eqn_config.dim_x = config.eqn_config.n_player*3
        config.eqn_config.dim_w = config.eqn_config.n_player*3
        config.eqn_config.dim_z = config.eqn_config.n_player*3

    return config
