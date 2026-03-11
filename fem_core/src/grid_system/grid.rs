use bevy::prelude::*;
use crate::grid_system::floor::Floor;

#[derive(Resource, Debug, Clone)]
pub struct Grid {
    pub origin: Vec3,                  // global origin
    pub spacing_x: f32,                // spacing in X
    pub spacing_y: f32,                // spacing in Y
    pub spacing_z: f32,                // floor height / Z spacing
    pub floors: Vec<Floor>,            // list of floors/levels
}

impl Grid {
    pub fn new(origin: Vec3, spacing_x: f32, spacing_y: f32, spacing_z: f32) -> Self {
        Self {
            origin,
            spacing_x,
            spacing_y,
            spacing_z,
            floors: Vec::new(),
        }
    }

    pub fn add_floor(&mut self, name: &str, level: f32) {
        self.floors.push(Floor::new(name, level));
    }

    pub fn get_floor(&self, name: &str) -> Option<&Floor> {
        self.floors.iter().find(|f| f.name == name)
    }

    pub fn num_floors(&self) -> usize {
        self.floors.len()
    }

    /// Convert grid coordinates (i,j,k) to global position
    pub fn grid_to_world(&self, i: usize, j: usize, floor_index: usize) -> Option<Vec3> {
        let floor = self.floors.get(floor_index)?;
        Some(Vec3::new(
            self.origin.x + i as f32 * self.spacing_x,
            floor.level,
            self.origin.z + j as f32 * self.spacing_y,
        ))
    }
}