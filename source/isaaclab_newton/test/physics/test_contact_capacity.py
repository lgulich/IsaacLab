# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Contact matching can enable sorting independently of the global determinism mode."""
from types import SimpleNamespace

import newton
import pytest
import warp as wp
from isaaclab.physics import PhysicsManager
from isaaclab_newton.physics import NewtonManager


@pytest.mark.parametrize('mode,matching', [
    (wp.DeterministicMode.NOT_GUARANTEED, 'latest'),
    (wp.DeterministicMode.GPU_TO_GPU, 'disabled'),
    (wp.DeterministicMode.NOT_GUARANTEED, 'disabled'),
])
def test_solver_capacity_growth_preserves_collision_pipeline(monkeypatch, mode, matching):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform((0, 0, .5), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=.5)
    model = builder.finalize(device='cpu')
    for key, value in {
        '_needs_collision_pipeline': True, '_collision_pipeline': None, '_contacts': None,
        '_model': model, '_deterministic_mode': mode,
        '_solver': SimpleNamespace(get_max_contact_count=lambda: 80),
        '_collision_cfg': SimpleNamespace(to_pipeline_args=lambda: {
            'rigid_contact_max': 32, 'contact_matching': matching,
        }),
    }.items():
        monkeypatch.setattr(NewtonManager, key, value)
    monkeypatch.setattr(PhysicsManager, '_device', 'cpu')
    NewtonManager._initialize_contacts()
    pipeline, contacts = NewtonManager._collision_pipeline, NewtonManager._contacts
    assert contacts.rigid_contact_max == 80
    pipeline.collide(model.state(), contacts)
    if pipeline.deterministic:
        assert pipeline._sort_key_array.shape[0] == contacts.rigid_contact_max
