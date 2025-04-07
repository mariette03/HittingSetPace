use crate::instance::Instance;
use crate::solve::Status;

/// Solve the instance assuming it has bounded treewidth
fn solve_via_treewidth(instance: &Instance) -> Status {
    //TODO: Use a TW-approach, needs to compute tw before, then do a dp on that
    Status::Stop
}
