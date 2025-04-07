use crate::instance::Instance;
use crate::solve::Status;

/// Solve the instance assuming it is planar
fn solve_planar(instance: &Instance) ->  Status {
    //TODO: We assume the instance is planar (in a simple graph setting).
    // The implementation should better not rely on that
    Status::Stop
}
