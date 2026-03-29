from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Subset

from diffsmile.config import config
from diffsmile.gnot_lightning import GNOTDiffusionLightningModule, VolSurfacePointCloudDataModule, scale_tte_mesh
from diffsmile.helpers.evaluation import save_buffers_raw
from diffsmile.helpers.plotter import calculate_and_plot_physics_metrics
from diffsmile.helpers.scenario_plotter import aggregate_and_plot
from diffsmile.model.loss import ButterflyArbitrageLoss, CalendarArbitrageLoss

mpl.use("Agg")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone GNOT validation inference loop and plot aggregates.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=Path(
            "/cluster/project/math/andtran/develop/masters_thesis/code/lightning_logs/version_110/checkpoints/gnot-best-epoch=464-val_loss=0.002852.ckpt"
        ),
        help="Path to lightning checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/cluster/project/math/andtran/develop/masters_thesis/code/data/conditional"),
        help="Directory containing spx_iv_dataset_full_365.pt and conditioning vectors.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gnot_eval"), help="Output directory.")
    parser.add_argument("--days-per-batch", type=int, default=4, help="Validation days loaded together in one outer batch.")
    parser.add_argument("--samples-per-day", type=int, default=8, help="Monte-Carlo samples generated per day.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--num-steps", type=int, default=-1, help="Sampling steps; -1 uses checkpoint scheduler T_1.")
    parser.add_argument("--max-val-days", type=int, default=0, help="If >0, only run the first N validation days.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
    parser.add_argument("--save-buffers", action="store_true", help="Save preds/gt/day arrays as compressed npz.")
    parser.add_argument("--buffers-name", type=str, default="inference_results.npz", help="Output npz filename.")
    parser.add_argument("--plot-name", type=str, default="aggregate_plot_new.png", help="Output plot filename.")
    parser.add_argument(
        "--physics-plot-name",
        type=str,
        default="physics_metrics_plot_new.png",
        help="Output filename for reconstruction/arbitrage metrics plot.",
    )
    parser.add_argument(
        "--physics-separate-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate and save 3 separate physics plots (MSE, calendar arb, butterfly arb). Use --no-physics-separate-plots to save one combined plot.",  # noqa: E501
    )
    parser.add_argument(
        "--physics-mse-plot-name",
        type=str,
        default="time_reconstr_error.png",
        help="Output filename for MSE time plot when --physics-separate-plots is set.",
    )
    parser.add_argument(
        "--physics-calendar-plot-name",
        type=str,
        default="time_calendar_arb.png",
        help="Output filename for calendar arbitrage plot when --physics-separate-plots is set.",
    )
    parser.add_argument(
        "--physics-butterfly-plot-name",
        type=str,
        default="time_butterfly_arb.png",
        help="Output filename for butterfly arbitrage plot when --physics-separate-plots is set.",
    )

    parser.add_argument("--idx-short", type=int, default=1)
    parser.add_argument("--idx-mid", type=int, default=8)
    parser.add_argument("--idx-long", type=int, default=18)
    parser.add_argument("--idx-atm", type=int, default=16)
    parser.add_argument("--date-fmt", type=str, default="%Y-%m")
    parser.add_argument(
        "--date-tick-interval-months",
        type=int,
        default=6,
        help="Month interval for date ticks in aggregate and physics metrics plots.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            msg = "CUDA requested but not available."
            raise RuntimeError(msg)
        return torch.device("cuda")
    if device_arg == "mps":
        if not torch.backends.mps.is_available():
            msg = "MPS requested but not available."
            raise RuntimeError(msg)
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:  # noqa: PLR0915, PLR0912, C901
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = _resolve_device(args.device)
    print(f"Using device: {device}")

    dm = VolSurfacePointCloudDataModule(
        data_dir=args.data_dir,
        batch_size=args.days_per_batch,
        num_workers=args.num_workers,
        config=config,
    )
    dm.setup()

    assert dm.val_dataset is not None
    assert dm.log_moneyness_grid is not None
    assert dm.ttm_grid is not None
    assert dm.data_mean is not None
    assert dm.data_std is not None

    arb_loss = CalendarArbitrageLoss(maturities=dm.ttm_grid.to(device))
    butterfly_loss = ButterflyArbitrageLoss(log_moneyness=dm.log_moneyness_grid.to(device), maturities=dm.ttm_grid.to(device))

    dataset_val = dm.val_dataset
    if args.max_val_days > 0:
        max_days = min(args.max_val_days, len(dataset_val))
        dataset_for_loader = Subset(dataset_val, list(range(max_days)))
    else:
        dataset_for_loader = dataset_val

    model = GNOTDiffusionLightningModule.load_from_checkpoint(str(args.checkpoint_path), strict=False)
    model.to(device)
    model.eval()
    scheduler = model.scheduler.to(device)

    H, W = config.IMAGE_HEIGHT, config.IMAGE_WIDTH
    N = H * W

    log_moneyness_grid = dm.log_moneyness_grid
    ttm_grid = dm.ttm_grid
    logk_mesh, tte_mesh = torch.meshgrid(log_moneyness_grid, ttm_grid, indexing="ij")
    tte_mesh = scale_tte_mesh(tte_mesh)

    base_coords = torch.stack([tte_mesh.flatten(), logk_mesh.flatten()], dim=1).to(device)

    day_loader = DataLoader(
        dataset_for_loader,
        batch_size=args.days_per_batch,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    std_gpu = dm.data_std.to(device)
    mean_gpu = dm.data_mean.to(device)

    preds_buffer: list = []
    gt_buffer: list = []
    day_indices_buffer: list[int] = []

    num_steps = scheduler.T_1 if args.num_steps <= 0 else args.num_steps
    if num_steps > scheduler.T_1:
        msg = f"num_steps={num_steps} exceeds scheduler.T_1={scheduler.T_1}"
        raise ValueError(msg)

    with torch.no_grad():
        for batch_idx, (surf_chunk, ema_short_chunk, ema_long_chunk, scalars_chunk, next_surf_chunk) in enumerate(
            tqdm.tqdm(day_loader, desc="Sampling")
        ):
            surf_chunk_gpu = surf_chunk.to(device, non_blocking=True)
            ema_short_chunk_gpu = ema_short_chunk.to(device, non_blocking=True)
            ema_long_chunk_gpu = ema_long_chunk.to(device, non_blocking=True)
            scalars_chunk_gpu = scalars_chunk.to(device, non_blocking=True)
            next_surf_chunk_gpu = next_surf_chunk.to(device, non_blocking=True)

            current_days = surf_chunk_gpu.size(0)
            total_items = current_days * args.samples_per_day

            surf_batch = surf_chunk_gpu.repeat_interleave(args.samples_per_day, dim=0)
            ema_short_batch = ema_short_chunk_gpu.repeat_interleave(args.samples_per_day, dim=0)
            ema_long_batch = ema_long_chunk_gpu.repeat_interleave(args.samples_per_day, dim=0)
            scalars_batch = scalars_chunk_gpu.repeat_interleave(args.samples_per_day, dim=0)
            next_surf_batch = next_surf_chunk_gpu.repeat_interleave(args.samples_per_day, dim=0)

            batch_coords = base_coords.unsqueeze(0).expand(total_items, -1, -1)

            surf_flat = surf_batch.permute(0, 2, 3, 1).reshape(total_items, N, 1)
            ema_short_flat = ema_short_batch.permute(0, 2, 3, 1).reshape(total_items, N, 1)
            ema_long_flat = ema_long_batch.permute(0, 2, 3, 1).reshape(total_items, N, 1)
            context_values = torch.cat([surf_flat, ema_short_flat, ema_long_flat], dim=-1)

            mask = torch.ones((total_items, 1, H, W), device=device)
            t_start = torch.full((total_items,), num_steps - 1, device=device, dtype=torch.long)

            initial_signal, initial_noise = scheduler.get_sampling_scales(t_start, mask)
            x_t = initial_signal * next_surf_batch + initial_noise * torch.randn_like(next_surf_batch)

            for t_ in reversed(range(num_steps)):
                t = torch.full((total_items,), t_, device=device, dtype=torch.long)

                _, noise_scale_img = scheduler.get_sampling_scales(t, mask)

                noisy_values_flat = x_t.permute(0, 2, 3, 1).reshape(total_items, N, 1)
                noise_scale_flat = noise_scale_img.permute(0, 2, 3, 1).reshape(total_items, N, 1)

                noise_pred_flat = model.model(
                    query_coords=batch_coords,
                    noisy_values=noisy_values_flat,
                    noise_scale=noise_scale_flat,
                    scalars=scalars_batch,
                    context_coords=batch_coords,
                    context_values=context_values,
                )

                noise_pred_img = noise_pred_flat.view(total_items, H, W, 1).permute(0, 3, 1, 2)
                x_t = scheduler.backward_eps(x_t=x_t, pred_eps=noise_pred_img, t=t, mask=mask)

            x_t = x_t.mul_(std_gpu).add_(mean_gpu).exp_()
            next_surf_batch = next_surf_batch.mul_(std_gpu).add_(mean_gpu).exp_()

            generated_flat = x_t.cpu().numpy().squeeze(1)
            gt_flat = next_surf_batch.cpu().numpy().squeeze(1)

            generated_structured = generated_flat.reshape(current_days, args.samples_per_day, H, W)
            gt_structured = gt_flat.reshape(current_days, args.samples_per_day, H, W)[:, 0, :, :]

            preds_buffer.extend([generated_structured[i] for i in range(current_days)])
            gt_buffer.extend([gt_structured[i] for i in range(current_days)])

            start_day = batch_idx * args.days_per_batch
            day_indices_buffer.extend(range(start_day, start_day + current_days))

    if args.save_buffers:
        save_buffers_raw(
            preds_buffer,
            gt_buffer,
            day_indices_buffer,
            filename=str(args.output_dir / args.buffers_name),
        )

    indices = {
        "SHORT": args.idx_short,
        "MID": args.idx_mid,
        "LONG": args.idx_long,
        "ATM": args.idx_atm,
    }

    day_to_date = {i: pd.to_datetime(dataset_val.dates[i + 1]) for i in range(len(gt_buffer))}

    figures_before = set(plt.get_fignums())
    aggregate_and_plot(
        preds_buffer,
        gt_buffer,
        day_indices_buffer,
        indices,
        day_to_date=day_to_date,  # type: ignore[arg-type]
        date_fmt=args.date_fmt,
        date_tick_interval_months=args.date_tick_interval_months,
    )

    new_figures = [n for n in plt.get_fignums() if n not in figures_before]
    if len(new_figures) == 0 and len(plt.get_fignums()) > 0:
        new_figures = [plt.get_fignums()[-1]]

    if len(new_figures) > 0:
        plot_path = args.output_dir / args.plot_name
        plt.figure(new_figures[-1]).savefig(plot_path, dpi=220, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")
    else:
        print("No matplotlib figure found after aggregate_and_plot().")

    figures_before = set(plt.get_fignums())
    calculate_and_plot_physics_metrics(
        preds_buffer,
        gt_buffer,
        day_indices_buffer,
        arb_loss,
        butterfly_loss,
        device=str(device),
        separate_plots=args.physics_separate_plots,
        day_to_date=day_to_date,  # type: ignore[arg-type]
        date_fmt=args.date_fmt,
        date_tick_interval_months=args.date_tick_interval_months,
    )

    new_figures = [n for n in plt.get_fignums() if n not in figures_before]
    if len(new_figures) == 0 and len(plt.get_fignums()) > 0:
        new_figures = [plt.get_fignums()[-1]]

    if len(new_figures) > 0:
        if args.physics_separate_plots:
            figure_ids = sorted(new_figures)
            target_names = [
                args.physics_mse_plot_name,
                args.physics_calendar_plot_name,
                args.physics_butterfly_plot_name,
            ]

            saved_count = 0
            for fig_id, file_name in zip(figure_ids, target_names, strict=False):
                output_path = args.output_dir / file_name
                plt.figure(fig_id).savefig(output_path, dpi=220, bbox_inches="tight")
                print(f"Saved physics metrics plot to {output_path}")
                saved_count += 1

            if saved_count < len(target_names):
                print(f"Warning: Expected 3 separate physics figures but found {len(figure_ids)}.")
        else:
            physics_plot_path = args.output_dir / args.physics_plot_name
            plt.figure(new_figures[-1]).savefig(physics_plot_path, dpi=220, bbox_inches="tight")
            print(f"Saved physics metrics plot to {physics_plot_path}")
    else:
        print("No matplotlib figure found after calculate_and_plot_physics_metrics().")


if __name__ == "__main__":
    main()
