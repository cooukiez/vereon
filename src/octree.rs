use glam::Vec3;
use log::info;
use crate::CHILD_OFFSET;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};

pub trait Octant {
    fn set_child(&self, child: u32) -> u32;
    fn check_child(&self, child: u32) -> bool;
    fn has_children(&self) -> bool;
    fn set_first_child_index(&self, index: u32) -> u32;
    fn first_child_index(&self) -> u32;
    fn child_mask(&self) -> u8;
    fn set_child_mask(&self, mask: u8) -> u32;
    fn child_count(&self) -> u32;
}

impl Octant for u32 {
    fn set_child(&self, child: u32) -> u32 {
        self | 1u32 << (child + CHILD_OFFSET)
    }

    fn check_child(&self, child: u32) -> bool {
        self & (1u32 << (child + CHILD_OFFSET)) > 0
    }

    fn has_children(&self) -> bool {
        self & 0b11111111_00000000_00000000_00000000 > 0
    }

    fn set_first_child_index(&self, index: u32) -> u32 {
        (self & 0b11111111_00000000_00000000_00000000)
            | (index & 0b00000000_11111111_11111111_11111111)
    }

    fn first_child_index(&self) -> u32 {
        self & 0b00000000_11111111_11111111_11111111
    }

    fn child_mask(&self) -> u8 {
        ((self & 0b11111111_00000000_00000000_00000000) >> CHILD_OFFSET) as u8
    }

    fn set_child_mask(&self, mask: u8) -> u32 {
        (self & 0b00000000_11111111_11111111_11111111)
            | (((mask as u32) << CHILD_OFFSET) & 0b11111111_00000000_00000000_00000000)
    }

    fn child_count(&self) -> u32 {
        (self >> CHILD_OFFSET).count_ones()
    }
}

pub fn encode_node(child_mask: u8, first_child_index: u32) -> u32 {
    ((child_mask as u32) << CHILD_OFFSET)
        | (first_child_index & 0b00000000_11111111_11111111_11111111)
}

pub struct SVO {
    pub nodes: Vec<u32>,
    pub root_span: f32,
    pub depth: u8,
}

impl SVO {
    pub fn new(depth: u8) -> Self {
        SVO {
            // insert root
            nodes: Vec::from([0]),
            root_span: 2u32.pow(depth as u32) as f32,
            depth,
        }
    }

    pub fn gen_random_svo(&mut self, seed: u64) {
        let mut rng = StdRng::seed_from_u64(seed);

        self.gen_random_branch(&mut rng, 0, 0); // start at root index & depth
    }

    pub fn gen_random_branch(&mut self, rng: &mut StdRng, cur_index: usize, cur_depth: u8) {
        if cur_depth < self.depth {
            let child_mask = rng.gen::<u8>();
            let first_child_index = self.nodes.len() as u32;
            let node = encode_node(child_mask, first_child_index);
            self.nodes[cur_index] = node;

            for i in 0..8 {
                self.nodes.push(0);

                if node.check_child(i) {
                    self.gen_random_branch(rng, (first_child_index + i) as usize, cur_depth + 1);
                }
            }
        }
    }

    pub fn insert_node_at_depth(&mut self, pos: Vec3, depth: u8) -> usize {
        let mut cs = self.root_span; // span
        let mut cd = 0; // depth
        let mut node_idx = 0;

        while cd < depth {
            cs *= 0.5;
            let mut child_idx = 0;
            if cs < pos.x { child_idx += 1; }
            if cs < pos.y { child_idx += 2; }
            if cs < pos.z { child_idx += 4; }

            if !self.nodes[node_idx].has_children() {
                self.nodes[node_idx] = self.nodes[node_idx].set_first_child_index(self.nodes.len() as u32);
                for _ in 0..8 { self.nodes.push(0); }
            }

            self.nodes[node_idx] = self.nodes[node_idx].set_child(child_idx);
            node_idx = (self.nodes[node_idx].first_child_index() + child_idx) as usize;
            cd += 1;
        }

        node_idx
    }

    pub fn insert_node(&mut self, pos: Vec3) -> usize {
        self.insert_node_at_depth(pos, self.depth)
    }

    pub fn count_notes(&self) -> u32 {
        self.nodes.iter().map(|&n| n > 1).count() as u32
    }
    
    pub fn log_octree(&self) {
        info!("svo, depth: {}, root_span: {}", self.depth, self.root_span);
        for (i, node) in self.nodes.iter().enumerate() {
            info!("{i}: cm: {:#08b}, fci: {:?}", node.child_mask(), node.first_child_index());
        }
    }
}   