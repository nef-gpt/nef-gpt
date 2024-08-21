# Evaluate MLP3D models generated by the GPT model.
# Visualize the models in 3D.

import copy

from matplotlib import pyplot as plt
from models.mlp_models import MLP3D
from utils import sdf_meshing
import trimesh

from utils.hd_utils import render_mesh


def model_to_mesh(mlp: MLP3D, res=512):
    effective_file_name = "models/visualization/tmp.ply"
    v, f, sdf = sdf_meshing.create_mesh(
        mlp,
        effective_file_name,
        N=res,
        level=0 if mlp.output_type in ["occ", "logits"] else 0,
    )
    if "occ" in mlp.output_type or "logits" in mlp.output_type:
        tmp = copy.deepcopy(f[:, 1])
        f[:, 1] = f[:, 2]
        f[:, 2] = tmp

    mesh = trimesh.Trimesh(v, f)
    return mesh, sdf


def visualize_model3d(mlp: MLP3D):

    mesh, sdf = model_to_mesh(mlp)

    mesh.show()
    # img, _ = render_mesh(mesh)
    # plt.imshow(img)
    # plt.show()


if __name__ == "__main__":

    # shapeNetData = ShapeNetDataset(
    #     "./datasets/plane_mlp_weights", transform=ModelTransform3D()
    # )

    # load first model and visualize
    # visualize_model3d(shapeNetData[4][0])
    pass
