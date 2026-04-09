"""GNME SDP backend exports.

This subpackage intentionally keeps imports lazy to avoid circular imports with
the vendored inflation modules.
"""

DEFAULT_GNME_SDP_BACKEND = "task"


def build_default_feasibility_model(*args, **kwargs):
    from .GNMETaskStateSDP import build_block_task_feasibility_model

    return build_block_task_feasibility_model(*args, **kwargs)


def build_fusion_feasibility_model(*args, **kwargs):
    from .GNMEStateSDP import build_fusion_feasibility_model as _impl

    return _impl(*args, **kwargs)


def build_sdp_draft(*args, **kwargs):
    from .GNMEStateSDP import build_sdp_draft as _impl

    return _impl(*args, **kwargs)


def build_top_down_sdp_draft(*args, **kwargs):
    from .GNMEStateSDP import build_top_down_sdp_draft as _impl

    return _impl(*args, **kwargs)


def build_smaller_sdp_draft(*args, **kwargs):
    from .GNMEStateSDP import build_smaller_sdp_draft as _impl

    return _impl(*args, **kwargs)


def build_top_down_feasibility_model(*args, **kwargs):
    from .GNMETaskStateSDP import build_top_down_block_task_feasibility_model as _impl

    return _impl(*args, **kwargs)


__all__ = [
    "DEFAULT_GNME_SDP_BACKEND",
    "build_default_feasibility_model",
    "build_fusion_feasibility_model",
    "build_sdp_draft",
    "build_top_down_sdp_draft",
    "build_smaller_sdp_draft",
    "build_top_down_feasibility_model",
]
