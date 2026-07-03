"""
Focal-loss patch for Ultralytics classification branch.

Why: the 10 Track-6 classes include several visually-similar groups that
plain BCE tends to confuse under class imbalance:
  - Vehicle.Car vs Pickup Truck vs Van
  - Single Truck vs Combo Truck vs Heavy Duty Vehicle vs Trailer
  - Motorcycle vs Bicycle (small, thin objects, easy to confuse at range)
Focal loss down-weights easy/majority-class examples so gradient signal
concentrates on the hard, rare, and confusable classes.

Ultralytics' v8 detection loss (ultralytics.utils.loss.v8DetectionLoss)
builds its classification loss as BCEWithLogitsLoss. This helper swaps
that in-place for FocalLoss(BCEWithLogitsLoss(), gamma) — the same
mechanism Ultralytics itself used for the legacy `fl_gamma` hyperparameter
before it was removed from the CLI in favor of manual wiring.

NOTE: Ultralytics internals change between versions; this monkeypatch is
guarded with a try/except and prints a warning + falls back to plain BCE
rather than crashing the whole Hafnia training job if the internal API
has moved. Pin the ultralytics version in requirements.txt to reduce risk.
"""
import warnings


def apply_focal_loss(model, gamma: float = 1.5, alpha: float = 0.25):
    try:
        from ultralytics.utils.loss import FocalLoss
        import torch.nn as nn

        # Patch happens lazily: Ultralytics builds the loss object on the
        # first call to model.train() / model.model.init_criterion(). We
        # wrap init_criterion so every criterion instance gets the patch.
        orig_init_criterion = model.model.init_criterion

        def patched_init_criterion():
            criterion = orig_init_criterion()
            if hasattr(criterion, "bce"):
                criterion.bce = FocalLoss(nn.BCEWithLogitsLoss(reduction="none"), gamma=gamma)
                print(f"[focal_patch] applied FocalLoss(gamma={gamma}) to classification branch")
            else:
                warnings.warn("[focal_patch] criterion has no .bce attribute; "
                               "ultralytics internals may have changed, skipping patch")
            return criterion

        model.model.init_criterion = patched_init_criterion
    except Exception as e:  # noqa: BLE001 - defensive: never break training over this
        warnings.warn(f"[focal_patch] could not apply focal loss ({e}); using default BCE")
