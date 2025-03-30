from install import install_and_init

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

def init_AI():
    from model_sys import build_model, compile_model
    from Collector.collect_data import collect_data
    
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


def init():
    print("Loading configuration file (config.txt)...")
    config=load_config()
    
    print("Checking installation status...")

    status=None
    with open("ProgramData/did_install","r") as f:
        status=f.read()

    if status!="1":
        print("Only minimal environemnt is installed, starting install protocol...")
        install_and_init()

if __name__=="__main__":
    init()
