import sys

def read_uncrustify_config(filename):
    config = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Ignore empty lines and comments
                # Split on the first '=' character to handle inline comments
                key_value = line.split('=', 1)
                if len(key_value) == 2:
                    key, value = key_value
                    config[key.strip()] = value.split('#')[0].strip()
                elif len(key_value) == 1:
                    key = key_value[0]
                    config[key.strip()] = ''
                else:
                    print(f"Ignoring invalid line: {line}")
    return config

def compare_config_files(file1, file2):
    config1 = read_uncrustify_config(file1)
    config2 = read_uncrustify_config(file2)

    common_settings = {}
    different_settings = {}

    for key, value in config1.items():
        if key in config2:
            if config2[key] == value:
                common_settings[key] = value
            else:
                different_settings[key] = (value, config2[key])
        else:
            different_settings[key] = (value, None)

    for key, value in config2.items():
        if key not in config1:
            different_settings[key] = (None, value)

    return common_settings, different_settings

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_uncrustify_configs.py <file1> <file2>")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]

    common, different = compare_config_files(file1, file2)

    print("Common Settings:")
    for key, value in common.items():
        print(f"{key} = {value}")
    
    print("\nDifferent Settings:")
    for key, (value1, value2) in different.items():
        if value1 is not None and value2 is not None:
            print(f"{key}: {value1} != {value2}")
        # elif value1 is not None:
        #     print(f"{key}: {value1} (Only in {file1})")
        # elif value2 is not None:
        #     print(f"{key}: {value2} (Only in {file2})")
