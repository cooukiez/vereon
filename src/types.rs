use glam::{Mat4, UVec2, Vec2, Vec4};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
    pub proj_mat: [[f32; 4]; 4],
    pub cam_pos: [f32; 4],
    pub cam_dir: [f32; 4],
    pub cam_plane_u: [f32; 4],
    pub cam_plane_v: [f32; 4],
    pub res: [u32; 2],
    pub mouse: [f32; 2],
    pub time: u32,
    pub _padding: [u32; 3],
}

impl Default for Uniform {
    fn default() -> Self {
        Uniform {
            proj_mat: Mat4::IDENTITY.to_cols_array_2d(),
            cam_pos: Vec4::ZERO.to_array(),
            cam_dir: Vec4::ZERO.to_array(),
            cam_plane_u: Vec4::ZERO.to_array(),
            cam_plane_v: Vec4::ZERO.to_array(),
            res: UVec2::ZERO.to_array(),
            mouse: Vec2::ZERO.to_array(),
            time: 0,
            _padding: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
}

impl Vertex {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x2];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        uv: [1.0, 1.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        uv: [0.0, 1.0],
    },
];

pub const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];
