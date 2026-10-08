"""Pure tensor friction feedforward, independent of simulator initialization."""
import torch


def capture_assist_scale(previous,captured,dt_s,ramp_s):
    if dt_s<=0 or ramp_s<0:raise ValueError('invalid capture-assist timing')
    if ramp_s==0:return captured.to(previous.dtype)
    return torch.where(captured,(previous+dt_s/ramp_s).clamp(max=1.),0.)


def friction_feedforward(error, velocity, static, dynamic, viscous, *,
                         static_gain=.9, position_deadband=1e-4, velocity_threshold=.001,
                         effort_budget=None, assist_only_toward_target=False, moving_gain=1.):
    if not 0<=static_gain<=1 or not 0<=moving_gain<=1 or position_deadband<0 or velocity_threshold<=0:
        raise ValueError('invalid friction compensation parameters')
    rest=static_gain*static*error.sign()
    rest=torch.where(error.abs()>position_deadband,rest,0.)
    moving=moving_gain*(dynamic*velocity.sign()+viscous*velocity)
    effort=torch.where(velocity.abs()>velocity_threshold,moving,rest)
    if assist_only_toward_target:
        # Keep natural friction for braking after overshoot; never assist motion
        # away from the current position target or through its deadband.
        toward=(error*velocity>=0)|(velocity.abs()<=velocity_threshold)
        effort=torch.where(toward&(error.abs()>position_deadband),effort,0.)
    if effort_budget is not None:
        budget=effort_budget.clamp(min=0)
        effort=torch.minimum(torch.maximum(effort,-budget),budget)
    return effort
