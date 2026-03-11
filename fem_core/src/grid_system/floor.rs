#[derive(Debug, Clone)]
pub struct Floor {
    pub name: String,
    pub level: f32,       // elevation in meters
}

impl Floor {
    pub fn new(name: &str, level: f32) -> Self {
        Self {
            name: name.to_string(),
            level,
        }
    }
}