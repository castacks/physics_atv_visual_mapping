import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from IPython.display import clear_output


def load_model(backbone, project_root):
    overrides = ["val_dataloader.batch_size=1", f"project_root={project_root}"]
    if "radio" in backbone:
        overrides += ["backbone=radio"]
    overrides += [f"backbone.name={backbone}"]
    
    # Initialize Hydra manually for Jupyter Notebook
    if not GlobalHydra.instance().is_initialized():
        initialize(config_path="../config", version_base=None)

    # Load configuration and overrides elements
    cfg = compose(config_name="base", overrides=overrides)

    # Load Backbones
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = instantiate(cfg.backbone)
    backbone.to(device)

    # Load Model
    model = instantiate(cfg.model)
    model.cuda()
    model.eval()

    # Load checkpoint
    model.load_state_dict(torch.load(f"./output/jafar/{backbone.name}.pth", weights_only=False)["jafar"])
    clear_output()
    return model, backbone
