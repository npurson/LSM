import lpips
import torch
import torch.nn.functional as F
from dust3r.losses import *
from einops import rearrange
from large_spatial_model.utils.camera_utils import get_scaled_camera
from large_spatial_model.utils.cuda_splatting import DummyPipeline, render
from large_spatial_model.utils.gaussian_model import GaussianModel
from torchmetrics import Accuracy, JaccardIndex
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchvision.utils import save_image
from vggt.utils.geometry import unproject_depth_map_to_point_map


class KWRegr3D(Regr3D):
    def get_all_pts3d(self, gt1, gt2, pred1, pred2, dist_clip=None, **kwargs):
        return super().get_all_pts3d(gt1, gt2, pred1, pred2, dist_clip)

class L2Loss(LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance

class L1Loss(LLoss):
    """ Manhattan distance between 3d points """

    def distance(self, a, b):
        return torch.abs(a - b).mean()  # L1 distance


L2 = L2Loss()
L1 = L1Loss()


def rotation_6d_to_matrix(d6):
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def merge_and_split_predictions(*args):
    merged = {}
    for key in [
        'scales', 'rotations', 'covs', 'opacities', 'sh_coeffs', 'means', 'gs_feats'
    ]:
        merged_pred = torch.stack([pred[key] for pred in args], dim=1)
        merged_pred = rearrange(merged_pred, 'b v h w ... -> b (v h w) ...')
        merged[key] = merged_pred

    # Split along the batch dimension
    batch_size = next(iter(merged.values())).shape[0]
    split = [{key: value[i] for key, value in merged.items()} for i in range(batch_size)]

    return split


class GaussianLoss(MultiLoss):

    def __init__(self,
                 ssim_weight=0.2,
                 feature_loss_weight=0.2,
                 labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
        super().__init__()
        self.ssim_weight = ssim_weight
        self.feature_loss_weight = feature_loss_weight
        self.labels = labels
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        self.miou = JaccardIndex(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.accuracy = Accuracy(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.pipeline = DummyPipeline()
        # bg_color
        self.register_buffer('bg_color', torch.tensor([0.0, 0.0, 0.0]).cuda())

    def get_name(self):
        return f'GaussianLoss(ssim_weight={self.ssim_weight})'

    def compute_loss(self, gt1, gt2, pred1, pred2, target_view=None, model=None):
        # render images
        # 1. merge predictions
        pred = merge_and_split_predictions(pred1, pred2)

        """
        # 2. calculate optimal scaling
        pred_pts1 = pred1['means']
        pred_pts2 = pred2['means']
        # convert to camera1 coordinates
        # everything is normalized w.r.t. camera of view1
        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'].to(in_camera1.device))  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'].to(in_camera1.device))  # B,H,W,3
        scaling = find_opt_scaling(
            gt_pts1, gt_pts2, pred_pts1, pred_pts2, valid1=valid1, valid2=valid2)
        """

        # 3. render images(need gaussian model, camera, pipeline)
        rendered_images = []
        rendered_feats = []
        gt_images = []

        for i in range(len(pred)):
            # get gaussian model
            gaussians = GaussianModel.from_predictions(pred[i], sh_degree=3)
            # get camera
            # ref_camera_extrinsics = gt1['camera_pose'][i]
            target_view_list = [gt1, gt2, target_view]  # use gt1, gt2, and target_view
            for j in range(len(target_view_list)):
                # target_extrinsics = target_view_list[j]['camera_pose'][i]
                target_intrinsics = target_view_list[j]['camera_intrinsics'][i]
                target_extrinsics = target_view_list[j]['extrinsics'][i]  # actually camera pose
                target_extrinsics_ = target_extrinsics.clone()
                target_extrinsics_[:3, :3] = target_extrinsics[:3, :3].mT
                target_extrinsics_[:3, 3:4] = -target_extrinsics_[:3, :3] @ target_extrinsics[:3, 3:4]

                image_shape = target_view_list[j]['true_shape'][i]
                scale = 1  # scaling[i]
                camera = get_scaled_camera(None,
                                           target_extrinsics.detach(),
                                           target_intrinsics, scale,
                                           image_shape)
                # render(image and features)
                rendered_output = render(
                    camera,
                    gaussians,
                    self.pipeline,
                    self.bg_color,
                    intrinsics=target_intrinsics,
                    extrinsics=target_extrinsics_)
                rendered_images.append(rendered_output['render'])
                rendered_feats.append(rendered_output['feature_map'])
                gt_images.append(target_view_list[j]['img'][i] * 0.5 + 0.5)

        rendered_images = torch.stack(rendered_images, dim=0)
        rendered_images = rendered_images.squeeze(1).permute(0, 3, 1, 2)  # B, 3, H, W
        gt_images = torch.stack(gt_images, dim=0)
        rendered_feats = torch.stack(rendered_feats, dim=0)
        rendered_feats = rendered_feats.squeeze(1).permute(0, 3, 1, 2)  # B, d_feats, H, W
        rendered_feats = model.feature_expansion(rendered_feats)  # B, 512, H//2, W//2
        rendered_depths = torch.stack(rendered_depths, dim=0).squeeze(1).permute(0, 3, 1, 2)
        gt_feats = model.lseg_feature_extractor.extract_features(gt_images)  # B, 512, H//2, W//2

        image_loss = torch.abs(rendered_images - gt_images).mean()
        feature_loss = (1 - torch.nn.functional.cosine_similarity(
            rendered_feats, gt_feats, dim=1)).mean()
        loss = image_loss + self.feature_loss_weight * feature_loss

        return loss, {'image_loss': float(image_loss), 'feature_loss': float(feature_loss)}


class TestLoss(MultiLoss):

    def __init__(self,
                 pose_align_steps=False,
                 labels=['wall', 'floor', 'ceiling', 'chair', 'table', 'sofa', 'bed', 'other']):
        super().__init__()
        self.labels = labels
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
        self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
        self.miou = JaccardIndex(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.accuracy = Accuracy(
            num_classes=len(self.labels) + 1,
            task='multiclass',
            ignore_index=0)
        self.pipeline = DummyPipeline()
        # bg_color
        self.register_buffer('bg_color', torch.tensor([0.0, 0.0, 0.0]).cuda())

        self.pose_align_steps = pose_align_steps
        if self.pose_align_steps:
            self.pose_embeds = nn.Embedding(2, 9)  # num_views - 1, Delta positions (3D) + Delta rotations (6D)
            self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
            torch.nn.init.zeros_(self.pose_embeds.weight)

    def get_name(self):
        return f'TestLoss'

    def compute_loss(self, gt1, gt2, pred1, pred2, target_view=None, model=None, pose_deltas=None, evaluate=True):
        # render images
        # 1. merge predictions
        pred = merge_and_split_predictions(pred1, pred2)

        """
        # 2. calculate optimal scaling
        pred_pts1 = pred1['means']
        pred_pts2 = pred2['means']
        # convert to camera1 coordinates
        # everything is normalized w.r.t. camera of view1
        valid1 = gt1['valid_mask'].clone()
        valid2 = gt2['valid_mask'].clone()
        in_camera1 = inv(gt1['camera_pose'])
        gt_pts1 = geotrf(in_camera1, gt1['pts3d'].to(in_camera1.device))  # B,H,W,3
        gt_pts2 = geotrf(in_camera1, gt2['pts3d'].to(in_camera1.device))  # B,H,W,3
        scaling = find_opt_scaling(
            gt_pts1, gt_pts2, pred_pts1, pred_pts2, valid1=valid1, valid2=valid2)
        """

        # 3. render images(need gaussian model, camera, pipeline)
        rendered_images = []
        rendered_feats = []
        gt_images = []

        for i in range(len(pred)):
            # get gaussian model
            gaussians = GaussianModel.from_predictions(pred[i], sh_degree=3)
            # get camera
            # ref_camera_extrinsics = gt1['camera_pose'][i]
            target_view_list = [gt1, gt2, target_view]  # use gt1, gt2, and target_view
            for j in range(len(target_view_list)):
                # target_extrinsics = target_view_list[j]['camera_pose'][i]
                target_intrinsics = target_view_list[j]['camera_intrinsics'][i]
                target_extrinsics = target_view_list[j]['extrinsics'][i]  # actually camera pose
                target_extrinsics_ = target_extrinsics.clone()
                target_extrinsics_[:3, :3] = target_extrinsics[:3, :3].mT
                target_extrinsics_[:3, 3:4] = -target_extrinsics_[:3, :3] @ target_extrinsics[:3, 3:4]

                if pose_deltas is not None:
                    assert i == 0
                    pose_deltas_ = pose_deltas.weight[j - 1].unsqueeze(0)
                    dx, drot = pose_deltas_[..., :3], pose_deltas_[..., 3:]
                    rot = rotation_6d_to_matrix(
                        drot + self.identity.expand(pose_deltas_.size(0), -1)
                    )  # (..., 3, 3)
                    transform = torch.eye(4, device=pose_deltas_.device).repeat((pose_deltas_.size(0), 1, 1))
                    transform[..., :3, :3] = rot
                    transform[..., :3, 3] = dx
                    target_extrinsics_ = target_extrinsics_ @ transform.squeeze(0)

                image_shape = target_view_list[j]['true_shape'][i]
                scale = 1  # scaling[i]
                camera = get_scaled_camera(None,
                                           target_extrinsics.detach(),
                                           target_intrinsics, scale,
                                           image_shape)
                # render(image and features)
                rendered_output = render(camera, gaussians, self.pipeline, self.bg_color,
                                         intrinsics=target_intrinsics,
                                         extrinsics=target_extrinsics_)
                rendered_images.append(rendered_output['render'])
                rendered_feats.append(rendered_output['feature_map'])
                gt_images.append(target_view_list[j]['img'][i] * 0.5 + 0.5)

        rendered_images = torch.stack(rendered_images, dim=0)  # B, 3, H, W
        rendered_images = rendered_images.squeeze(1).permute(0, 3, 1, 2)
        gt_images = torch.stack(gt_images, dim=0)
        rendered_feats = torch.stack(rendered_feats, dim=0)  # B, d_feats, H, W
        rendered_feats = rendered_feats.squeeze(1).permute(0, 3, 1, 2)
        rendered_feats = model.feature_expansion(rendered_feats)  # B, 512, H//2, W//2
        gt_feats = model.lseg_feature_extractor.extract_features(gt_images)  # B, 512, H//2, W//2

        image_loss = torch.abs(rendered_images[-1] - gt_images[-1]).mean()
        feature_loss = (1 - torch.nn.functional.cosine_similarity(
            rendered_feats[-1], gt_feats[-1], dim=1)).mean()

        logits = model.lseg_feature_extractor.decode_feature(rendered_feats, self.labels)
        pred = logits.argmax(dim=1, keepdim=True)
        pred = pred.clamp(max=7) + 1
        pred = pred[-1]

        if evaluate:
            self.miou.update(pred, target_view["labelmap"].long())
            self.psnr.update(rendered_images[-1], gt_images[-1])

        loss = image_loss + feature_loss
        return loss, {'image_loss': float(image_loss), 'feature_loss': float(feature_loss)}


# loss for one batch
def loss_of_one_batch(batch,
                      model,
                      criterion,
                      device,
                      symmetrize_batch=False,
                      use_amp=False,
                      ret=None):
    view1, view2, target_view = batch
    ignore_keys = set([
        'depthmap', 'dataset', 'label', 'instance', 'idx', 'true_shape', 'rng'
    ])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    # if symmetrize_batch:
    #     view1, view2 = make_batch_symmetric((view1, view2))
    #     target_view, _ = make_batch_symmetric((target_view, target_view))
    # Get the actual model if it's distributed
    actual_model = model.module if hasattr(model, 'module') else model

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):
        if actual_model.training:
            pred1, pred2 = actual_model(view1, view2)
        else:
            with torch.no_grad():
                pred1, pred2 = actual_model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            pose_align_steps = getattr(criterion, 'pose_align_steps', 0)
            if pose_align_steps:
                optimzer = torch.optim.Adam(criterion.pose_embeds.parameters(), lr=5e-3)
                criterion.pose_embeds.weight.data.zero_()
                for i in range(pose_align_steps):
                    if i != pose_align_steps - 1:
                        loss, _ = criterion(
                            view1, view2, pred1, pred2, target_view=target_view, model=actual_model,
                            pose_deltas=criterion.pose_embeds, evaluate=False)
                        optimzer.zero_grad()
                        loss.backward()
                        optimzer.step()
                    else:
                        loss = criterion(
                            view1, view2, pred1, pred2, target_view=target_view, model=actual_model,
                            pose_deltas=criterion.pose_embeds)

            loss = criterion(
                view1, view2, pred1, pred2, target_view=target_view, model=actual_model
            ) if criterion is not None else None

    result = dict(
        view1=view1,
        view2=view2,
        target_view=target_view,
        pred1=pred1,
        pred2=pred2,
        loss=loss)
    return result[ret] if ret else result
