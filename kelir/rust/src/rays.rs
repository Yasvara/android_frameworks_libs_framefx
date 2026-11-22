// SPDX-FileCopyrightText: IRedDragonICY
// SPDX-License-Identifier: LicenseRef-IRedDragonICY-NC-1.0

//! What of the game's scene the ray-traced kinds trace against, frame by frame: which meshes get
//! a bottom-level acceleration structure built, which skinned ones are refitted, which instances
//! the top level holds, and which structures go to stay within memory. The recorder hands in the
//! meshes it saw drawn; the Vulkan side builds what the plan says. The budget limits instances,
//! builds and triangles, never rays.

use std::collections::HashMap;

pub const MAX_BUILDS: usize = 8;
pub const MAX_REFITS: usize = 16;
pub const MAX_INSTANCES: usize = 256;
pub const MAX_EVICTS: usize = 8;

/// A mesh drawn this frame. Mirrored by framefx::kelir::RayMesh.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct RayMesh {
    /// The mesh's content: buffer, range and layout hashed. Never 0.
    pub id: u64,
    pub triangles: u32,
    /// Drawn with a vertex shader that moves it (characters): refitted, not rebuilt.
    pub skinned: bool,
    /// Bounding sphere in world space.
    pub center: [f32; 3],
    pub radius: f32,
}

/// Mirrored by framefx::kelir::RayBudget.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RayBudget {
    pub camera: [f32; 3],
    /// Beyond this, a mesh is left to screen-space effects.
    pub cull_distance: f32,
    /// Radius over distance below which a mesh is too small to matter.
    pub min_size: f32,
    pub max_instances: u32,
    pub max_builds: u32,
    pub max_refits: u32,
    /// Triangles resident across all bottom-level structures.
    pub max_triangles: u32,
}

/// One frame's work. Mirrored by framefx::kelir::RayPlan.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct RayPlan {
    pub build_count: u32,
    pub builds: [u64; MAX_BUILDS],
    pub refit_count: u32,
    pub refits: [u64; MAX_REFITS],
    /// Indices into the frame's meshes, nearest-largest first.
    pub instance_count: u32,
    pub instances: [u32; MAX_INSTANCES],
    pub evict_count: u32,
    pub evicts: [u64; MAX_EVICTS],
}

impl Default for RayPlan {
    fn default() -> Self {
        RayPlan {
            build_count: 0,
            builds: [0; MAX_BUILDS],
            refit_count: 0,
            refits: [0; MAX_REFITS],
            instance_count: 0,
            instances: [0; MAX_INSTANCES],
            evict_count: 0,
            evicts: [0; MAX_EVICTS],
        }
    }
}

struct Resident {
    triangles: u32,
    last_used: u32,
}

/// The structures built so far, across frames.
#[derive(Default)]
pub struct RayScene {
    resident: HashMap<u64, Resident>,
    triangles: u64,
    frame: u32,
}

fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

impl RayScene {
    pub fn plan(&mut self, meshes: &[RayMesh], budget: &RayBudget) -> RayPlan {
        self.frame = self.frame.wrapping_add(1);
        let mut out = RayPlan::default();
        // In reach and large enough on screen, biggest first.
        let mut picked: Vec<(usize, f32, f32)> = meshes
            .iter()
            .enumerate()
            .filter(|(_, m)| m.id != 0 && m.triangles > 0)
            .filter_map(|(i, m)| {
                let d = distance(&m.center, &budget.camera);
                let size = m.radius / d.max(1e-3);
                (d - m.radius <= budget.cull_distance && size >= budget.min_size).then_some((i, size, d))
            })
            .collect();
        picked.sort_by(|a, b| b.1.total_cmp(&a.1));
        picked.truncate((budget.max_instances as usize).min(MAX_INSTANCES));

        let max_builds = (budget.max_builds as usize).min(MAX_BUILDS);
        let mut refit_order: Vec<(f32, u64)> = Vec::new();
        for &(i, _, d) in &picked {
            let m = &meshes[i];
            match self.resident.get_mut(&m.id) {
                Some(r) => {
                    r.last_used = self.frame;
                    if m.skinned {
                        refit_order.push((d, m.id));
                    }
                }
                None => {
                    if (out.build_count as usize) < max_builds {
                        out.builds[out.build_count as usize] = m.id;
                        out.build_count += 1;
                        self.resident.insert(m.id, Resident { triangles: m.triangles, last_used: self.frame });
                        self.triangles += m.triangles as u64;
                    } else {
                        continue; // built on a later frame; not traced until then
                    }
                }
            }
            out.instances[out.instance_count as usize] = i as u32;
            out.instance_count += 1;
        }
        // Characters nearest the camera keep their shape first.
        refit_order.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (_, id) in refit_order.into_iter().take((budget.max_refits as usize).min(MAX_REFITS)) {
            out.refits[out.refit_count as usize] = id;
            out.refit_count += 1;
        }
        // Over the triangle budget: the longest unused go, never one traced this frame.
        if self.triangles > budget.max_triangles as u64 {
            let mut idle: Vec<(u32, u64)> = self
                .resident
                .iter()
                .filter(|(_, r)| r.last_used != self.frame)
                .map(|(&id, r)| (r.last_used, id))
                .collect();
            idle.sort();
            for (_, id) in idle {
                if self.triangles <= budget.max_triangles as u64 || out.evict_count as usize == MAX_EVICTS {
                    break;
                }
                if let Some(r) = self.resident.remove(&id) {
                    self.triangles -= r.triangles as u64;
                    out.evicts[out.evict_count as usize] = id;
                    out.evict_count += 1;
                }
            }
        }
        out
    }

    /// The Vulkan side could not build `id` (out of memory): forget it so it is asked again.
    pub fn failed(&mut self, id: u64) {
        if let Some(r) = self.resident.remove(&id) {
            self.triangles -= r.triangles as u64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> RayBudget {
        RayBudget {
            camera: [0.0; 3],
            cull_distance: 100.0,
            min_size: 0.01,
            max_instances: 4,
            max_builds: 2,
            max_refits: 1,
            max_triangles: 10_000,
        }
    }

    fn mesh(id: u64, x: f32, radius: f32, triangles: u32, skinned: bool) -> RayMesh {
        RayMesh { id, triangles, skinned, center: [x, 0.0, 0.0], radius }
    }

    #[test]
    fn builds_within_budget_and_traces_only_what_is_built() {
        let mut scene = RayScene::default();
        let meshes = [mesh(1, 10.0, 2.0, 100, false), mesh(2, 20.0, 2.0, 100, false), mesh(3, 5.0, 1.0, 100, false)];
        let p = scene.plan(&meshes, &budget());
        assert_eq!(p.build_count, 2);
        assert_eq!(p.instance_count, 2);
        let p = scene.plan(&meshes, &budget());
        assert_eq!(p.build_count, 1);
        assert_eq!(p.instance_count, 3);
    }

    #[test]
    fn culls_far_and_tiny_meshes() {
        let mut scene = RayScene::default();
        let meshes = [mesh(1, 500.0, 2.0, 100, false), mesh(2, 50.0, 0.1, 100, false), mesh(3, 10.0, 1.0, 100, false)];
        let p = scene.plan(&meshes, &budget());
        assert_eq!(p.instance_count, 1);
        assert_eq!(p.instances[0], 2);
    }

    #[test]
    fn refits_the_nearest_character_first() {
        let mut scene = RayScene::default();
        let meshes = [mesh(1, 30.0, 3.0, 100, true), mesh(2, 8.0, 1.0, 100, true)];
        scene.plan(&meshes, &budget());
        let p = scene.plan(&meshes, &budget());
        assert_eq!(p.refit_count, 1);
        assert_eq!(p.refits[0], 2);
    }

    #[test]
    fn evicts_the_longest_unused_over_budget() {
        let mut scene = RayScene::default();
        let mut b = budget();
        b.max_triangles = 250;
        scene.plan(&[mesh(1, 10.0, 2.0, 100, false), mesh(2, 10.0, 2.0, 100, false)], &b);
        scene.plan(&[mesh(3, 10.0, 2.0, 100, false)], &b);
        let p = scene.plan(&[mesh(3, 10.0, 2.0, 100, false), mesh(4, 10.0, 2.0, 100, false)], &b);
        assert_eq!(p.evict_count, 1);
        assert!(p.evicts[0] == 1 || p.evicts[0] == 2);
    }
}
