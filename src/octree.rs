use glam::Vec3;
use log::info;
use vss_rs::svo::{Octant, SVO};
use vss_rs::vox::index_to_pos;
use crate::types::{Vertex, BASE_CUBE_IDX, BASE_CUBE_UV, BASE_CUBE_VERT};

pub trait SVOExt {
    fn log(&self);
    fn polygonize_branch(&self, max_depth: u32, cd: u32, cs: f32, ni: u32, pos: Vec3, vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>);
    fn polygonize(&self, max_depth: u32) -> (Vec<Vertex>, Vec<u32>);
}

impl SVOExt for SVO {
    fn log(&self) {
        info!("svo, depth: {}, root_span: {}", self.depth, self.root_span);
        for (i, node) in self.nodes.iter().enumerate() {
            info!("{i}: cm: {:#08b}, fci: {:?}", node.child_mask(), node.first_child_index());
        }
    }

    fn polygonize_branch(
        &self,
        max_depth: u32,
        cd: u32,
        cs: f32,
        ni: u32,
        pos: Vec3,
        vertices: &mut Vec<Vertex>,
        indices: &mut Vec<u32>,
    ) {
        let cd = cd + 1;

        let node = self.nodes[ni as usize];
        if cd < max_depth && node.has_children() {
            let cs = cs * 0.5;

            for ci in 0..8 {
                let pos = pos + index_to_pos(ci, 2).as_vec3() * cs;

                if node.check_child(ci) {
                    self.polygonize_branch(max_depth, cd, cs, node.first_child_index() + ci, pos, vertices, indices);
                }
            }
        } else if node > 0 {
            let center_pos = pos + Vec3::splat(cs * 0.25);
            let vertex_offset = vertices.len() as u32;

            for i in 0..BASE_CUBE_VERT.len() {
                vertices.push(Vertex {
                    position: (center_pos + Vec3::from(BASE_CUBE_VERT[i]) * cs).to_array(),
                    uv: [BASE_CUBE_UV[i].0, BASE_CUBE_UV[i].1],
                });

                indices.push(BASE_CUBE_IDX[i] + vertex_offset);
            }
        }
    }

    fn polygonize(&self, max_depth: u32) -> (Vec<Vertex>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        self.polygonize_branch(max_depth, 0, 1.0, 0, Vec3::ZERO, &mut vertices, &mut indices);

        (vertices, indices)
    }
}