use glam::{Mat4, Vec2, Vec3};

pub struct Camera {
    // settings
    pub mov_speed_slow: f32,
    pub mov_speed_fast: f32,

    pub fov: f32,
    pub sensitivity: f32,
    pub mouse_sign: Vec2,

    pub z_near: f32,
    pub z_far: f32,

    pub rotation_min: Vec2, // x: yaw, y: pitch
    pub rotation_max: Vec2, // x: yaw, y: pitch

    pub up_axis: Vec3,

    pub enable_matrices: bool,

    // internal
    pub rotation: Vec2, // x: yaw, y: pitch

    pub front: Vec3,
    pub right: Vec3,
    pub up: Vec3,

    pub pos: Vec3,

    pub mov_speed: f32,
    pub mov_lin: Vec3,
    pub mov_lat: Vec3,

    pub plane_u: Vec3,
    pub plane_v: Vec3,

    pub aspect_ratio: f32,
    pub fov_tan: f32, // has to be updated when fov changes

    pub proj: Mat4,
}

impl Camera {
    pub fn update_proj(&mut self, window_width: u32, window_height: u32) {
        self.aspect_ratio = (window_width as f32) / (window_height as f32);
        if self.enable_matrices {
            self.proj =
                Mat4::perspective_lh(self.fov_tan, self.aspect_ratio, self.z_near, self.z_far);
        }
    }

    pub fn rotate(&mut self, mouse_delta: Vec2) {
        self.rotation -= mouse_delta * self.sensitivity * self.mouse_sign;
        self.rotation = self.rotation.clamp(self.rotation_min, self.rotation_max);

        self.front = Vec3::new(
            self.rotation.x.to_radians().cos() * self.rotation.y.to_radians().cos(),
            self.rotation.y.to_radians().sin(),
            self.rotation.x.to_radians().sin() * self.rotation.y.to_radians().cos(),
        )
        .normalize();

        self.right = self.front.cross(self.up_axis).normalize();
        self.up = self.right.cross(self.front).normalize();

        self.mov_lin = self.front * self.mov_speed;
        self.mov_lat = self.right * self.mov_speed;

        self.plane_u = self.right * self.aspect_ratio * self.fov_tan;
        self.plane_v = self.up * self.aspect_ratio * self.fov_tan;
    }

    pub fn get_view_proj(&self) -> Mat4 {
        if self.enable_matrices {
            let view = Mat4::look_at_lh(self.pos, self.pos + self.front, self.up);
            self.proj * view
        } else {
            Mat4::IDENTITY
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        let mut cam = Camera {
            // settings
            mov_speed_slow: 0.05,
            mov_speed_fast: 0.1,

            fov: 60.0,
            sensitivity: 0.1,
            mouse_sign: Vec2::new(-1.0, 1.0),

            z_near: 0.1,
            z_far: 100.0,

            rotation_min: Vec2::new(-180.0, -89.0),
            rotation_max: Vec2::new(180.0, 89.0),

            up_axis: Vec3::new(0.0, 1.0, 0.0),

            enable_matrices: false,

            // internal
            rotation: Vec2::new(-90.0, 0.0),

            front: Vec3::ZERO,
            right: Vec3::ZERO,
            up: Vec3::ZERO,

            pos: Vec3::ZERO,

            mov_speed: 0.0,
            mov_lin: Vec3::ZERO,
            mov_lat: Vec3::ZERO,

            plane_u: Vec3::ZERO,
            plane_v: Vec3::ZERO,

            aspect_ratio: 0.0,
            fov_tan: 0.0,

            proj: Mat4::IDENTITY,
        };

        cam.fov_tan = (cam.fov / 2.0).to_radians().tan();
        cam.mov_speed = cam.mov_speed_slow;
        cam
    }
}
