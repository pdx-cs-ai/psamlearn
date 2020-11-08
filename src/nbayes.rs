use crate::{Model, Instance};

pub struct NBayes;

pub fn train(samples: &[&Instance]) -> Box<NBayes> {
    todo!()
}

impl Model for NBayes {
    fn classify(&self, t: &Instance) -> bool {
        todo!()
    }
}
