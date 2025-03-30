def load_config():
    config_raw=None

    with open("config.txt","r") as f:
        config_raw=f.read()
    
    config_raw=config_raw.split("\n")
    
    config={}

    for line in config_raw:
        line=line.strip()
        line=line.split(":")
        config[line[0]]=line[1]
    
    return config

def init():
    config=load_config()

    model=build_model(
        config["mod.conv_filters"].split(","),
        config["mod.sizes"].split(","),
        config["mod.conv_stridesX"].split(","),
        config["mod.conv_stridesY"].split(","),
        config["mod.conv_LReLU_alpha"].split(","),
        config["mod.conv_pool_sizes"].split(","),
        config["mod.dense_sizes"].split(","),
        config["mod.dense_activation_methods"].split(",")
    )

    compile_model(model)

    train_dataset, test_dataset=collect_data(config["pre.gdown_file_id"], config["pre.train_batch_size"], config["pre.test_batch_size"], config["pre.contrast_strength"])

    return (config, model, train_dataset, test_dataset)

def main():
    print("Initializing local side...")
    init_data=init()
    print("Local side initiated!")

if __name__=="__main__":
    main()
