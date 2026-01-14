# test_recipe_direct.py
from stofs3d_setup.config.schema import Settings
from stofs3d_setup.recipes.generic import build

cfg = Settings.from_yaml("configs/v7p3_2020_atlas.yml")
build(cfg)

pass
