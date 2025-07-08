# Code adapted from: https://github.com/hubertrybka/vae-annealing/blob/main/annealing.py
#
import math
from scipy.special import expit  # numerically stable sigmoid
import torch # Added torch import for state_dict


################################
# KL-Annealer
################################
class Annealer:
    """
    This class is used to provide an annealing weight or value over the course of training.
    Call `step()` after each optimization step to update the internal counter.
    Call `get_weight()` or `get_weight_at(step)` to get the current or specific annealing value [0-1].
    Call `__call__(kld)` to get the annealed KLD value (scaled by the weight).
    """

    def __init__(self, total_steps=10000, shape='cosine', baseline=0.0, cyclical=False, disable=False, reverse=False):
        """
        Parameters:
            total_steps (int): Number of steps over which the primary annealing phase occurs.
                               For cyclical annealing, this is the duration of one cycle segment (e.g., up or down).
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
                              The annealing goes from `baseline` to 1.0.
            cyclical (bool): If True, the annealing repeats after `total_steps` is reached.
            disable (bool): If true, `__call__` and `get_weight` methods return 1.0.
            reverse (bool): If true, the annealing goes from 1.0 down to `baseline`.
        """

        self.current_step = 0
        self.reverse = reverse

        if shape not in ['linear', 'cosine', 'logistic']:
            raise ValueError(f"Shape must be one of 'linear', 'cosine', or 'logistic.', got {shape}")
        self.shape = shape

        if not 0 <= float(baseline) <= 1:
            raise ValueError(f"Baseline must be a float between 0 and 1, got {baseline}")
        self.baseline = float(baseline)

        if type(total_steps) is not int or total_steps < 1:
            raise ValueError(f"Argument total_steps must be an integer greater than 0, got {total_steps}")
        self.total_steps = total_steps

        if type(cyclical) is not bool:
            raise ValueError("Argument cyclical must be a boolean.")
        self.cyclical = cyclical

        if type(disable) is not bool:
            raise ValueError("Argument disable must be a boolean.")
        self.disable = disable

    def __call__(self, kld):
        """
        Scales the input KLD value by the current annealing weight.
        Args:
            kld (torch.Tensor): KL divergence loss
        Returns:
            out (torch.Tensor): KL divergence loss multiplied by the current annealing weight.
        """
        if self.disable:
            if isinstance(kld, torch.Tensor):
                 return kld
            else:
                 return kld * 1.0
        return kld * self.get_weight()

    def step(self):
        """
        Increments the internal step counter. If cyclical, wraps around after total_steps.
        """
        if self.disable:
            return # Do not step if disabled

        self.current_step += 1
        if self.cyclical and self.total_steps > 0 and self.current_step >= self.total_steps:
            self.current_step %= self.total_steps # Wrap around for cyclical

    def get_weight(self):
        """
        Returns the current annealing weight [baseline, 1.0].
        """
        if self.disable:
            return 1.0
        return self.get_weight_at(self.current_step)

    def get_weight_at(self, step):
        """
        Returns the annealing weight [baseline, 1.0] at a specific step.
        Args:
            step (int): The step at which to calculate the weight.
        """
        if self.disable:
            return 1.0

        step_for_calc = step
        if self.total_steps > 0:
            if self.cyclical:
                 step_for_calc = step % self.total_steps
            else:
                 step_for_calc = min(step, self.total_steps)

            progress = step_for_calc / self.total_steps # Progress within the segment [0, 1]
        else:
             progress = 0.0 # Handle total_steps = 0 case, though init prevents this


        if self.shape == 'linear':
            y = progress
        elif self.shape == 'cosine':
            y = (math.cos(math.pi * (progress - 1)) + 1) / 2
        elif self.shape == 'logistic':
             logistic_input = (progress * 12) - 6
             y = expit(logistic_input)
        else:
            y = 1.0 # Should not happen

        if self.reverse:
            y = 1.0 - y

        y_out = y * (1.0 - self.baseline) + self.baseline
        return y_out

    def state_dict(self):
        """Returns the state of the annealer as a dict."""
        return {
            'current_step': self.current_step,
        }

    def load_state_dict(self, state_dict):
        """Loads the annealer state from a dict."""
        self.current_step = state_dict['current_step']


################################
# Beta Annealer
################################
class BetaAnnealer:
    """
    Anneals the beta coefficient for the KL divergence loss, allowing for
    warmup, annealing cycles (standard up/down or Fu et al. ramp-up-hold),
    and cooldown phases.
    Manages a base Annealer internally for the shape of the annealing segments.
    """
    def __init__(self, max_beta=4.0, shape='cosine', baseline=0.0, reverse=False,
                 warmup_steps=0, warmup_value=1e-4, cooldown_steps=0, cooldown_value=None,
                 # --- Cycle Mode Parameters ---
                 cycle_mode='standard', num_cycles=1, # Overall cycle control
                 # Standard mode specific
                 total_steps=10000, mirror=False, # Use total_steps as segment_duration for standard
                 # Fu et al. mode specific
                 cycle_length=None, ramp_ratio=0.5 # Use cycle_length and ramp_ratio for Fu et al.
                 ):
        """
        Parameters:
            max_beta (float): The maximum beta value reached during annealing.
            shape (str): Shape of the annealing curve within a segment ('linear', 'cosine', 'logistic').
                         Applies to the ramp part of Fu et al. mode or the segment in standard mode.
            baseline (float): Starting weight [0-1] for the first 'up' segment (multiplied by max_beta in standard mode).
                              For Fu et al. mode, this affects the start of the ramp (beta starts at `baseline * max_beta`).
            reverse (bool): If True, the segments in standard mode anneal from max_beta down to baseline*max_beta (or vice versa if mirror=True).
                            For Fu et al. mode, if applied, would reverse the ramp shape (e.g., ramp down), less common.
            warmup_steps (int): Number of steps at the start with a fixed `warmup_value`.
            warmup_value (float): The beta value during the warmup phase.
            cooldown_steps (int): Number of steps at the end with a fixed `cooldown_value`. Cooldown occurs after all cycles.
            cooldown_value (float, optional): The beta value during the cooldown phase. Defaults to `warmup_value`.

            # --- Cycle Mode Parameters ---
            cycle_mode (str): 'standard' (current behavior: up or up/down) or 'fu_et_al' (ramp-up-hold).
            num_cycles (int): Total number of cycles. Must be > 0. Default 1.

            # Standard mode specific (used if cycle_mode='standard')
            total_steps (int): Duration of one 'up' or 'down' segment within a standard cycle. Default 10000.
            mirror (bool): If True, each standard cycle includes an 'up' phase followed by a 'down' phase, each of duration `total_steps`. Default False.

            # Fu et al. mode specific (used if cycle_mode='fu_et_al')
            cycle_length (int): Total steps in one Fu et al. cycle (ramp + hold). Required if cycle_mode='fu_et_al'.
            ramp_ratio (float): Ratio of cycle_length used for the ramp-up phase [0-1]. Default 0.5.
        """
        self.max_beta = max_beta
        self.shape = shape
        self.baseline = float(baseline)
        self.reverse_segment = bool(reverse) # Applies to the base shape progression

        self.warmup_steps = max(0, int(warmup_steps))
        self.warmup_value = float(warmup_value)

        self.cooldown_steps = max(0, int(cooldown_steps))
        self.cooldown_value = float(cooldown_value) if cooldown_value is not None else self.warmup_value

        self.cycle_mode = cycle_mode.lower()
        valid_cycle_modes = ['standard', 'fu_et_al']
        if self.cycle_mode not in valid_cycle_modes:
             raise ValueError(f"Invalid cycle_mode '{cycle_mode}'. Must be one of {valid_cycle_modes}")

        self.num_cycles = max(1, int(num_cycles)) # Must be at least 1 cycle

        # Determine cycle parameters based on mode
        if self.cycle_mode == 'standard':
            # 'total_steps' parameter from init is used as segment_duration
            self.segment_duration = max(1, int(total_steps))
            self.mirror = bool(mirror)
            self.cycle_length = self.segment_duration * (2 if self.mirror else 1) # Length of a full standard cycle
            # Ramp duration for the internal annealer is the segment duration
            ramp_duration_for_annealer = self.segment_duration
            self.ramp_ratio = 1.0 if not self.mirror else 0.5 # Ramp covers full segment if not mirrored

        elif self.cycle_mode == 'fu_et_al':
            # 'cycle_length' parameter from init is used
            if cycle_length is None or int(cycle_length) < 1:
                 raise ValueError("cycle_length must be provided and positive for 'fu_et_al' cycle_mode.")
            self.cycle_length = max(1, int(cycle_length))
            self.ramp_ratio = float(ramp_ratio)
            if not 0 <= self.ramp_ratio <= 1:
                 raise ValueError("ramp_ratio must be between 0 and 1 for 'fu_et_al' cycle_mode.")
            # Ramp duration for the internal annealer is cycle_length * ramp_ratio
            self.segment_duration = int(self.cycle_length * self.ramp_ratio) # segment_duration is the ramp part
            # Ensure ramp duration is at least 1 if cycle_length and ramp_ratio are positive
            self.segment_duration = max(1, self.segment_duration)
            ramp_duration_for_annealer = self.segment_duration
            self.mirror = False # Mirror concept doesn't apply in fu_et_al mode


        # The internal annealer handles the shape from baseline to 1.0 over its total_steps (the ramp duration)
        # It is NOT cyclical. BetaAnnealer handles the overall cycling.
        # reverse=False here means the *base* shape goes from baseline to 1.0.
        # BetaAnnealer's reverse_segment will handle reversing the final beta value range.
        self.annealer = Annealer(total_steps=ramp_duration_for_annealer, shape=self.shape, baseline=self.baseline, cyclical=False, disable=False, reverse=False)


        self.current_step = 0 # Total steps taken since start

        # Calculate phase durations
        self.warmup_end = self.warmup_steps
        self.annealing_duration_total = self.cycle_length * self.num_cycles
        self.annealing_end = self.warmup_end + self.annealing_duration_total
        self.total_duration_steps = self.annealing_end + self.cooldown_steps


    def __call__(self):
        """
        Returns the current beta coefficient based on the current step and phase.
        """
        if self.current_step < self.warmup_end:
            # Warmup phase
            return self.warmup_value
        elif self.current_step < self.annealing_end:
            # Annealing cycles phase
            step_in_annealing = self.current_step - self.warmup_end # Step count within the total annealing duration
            step_in_cycle = step_in_annealing % self.cycle_length   # Step count within the current cycle

            if self.cycle_mode == 'standard':
                # Standard mode (up or up/down)
                # Step within the current segment (up or down)
                step_in_segment = step_in_cycle % self.segment_duration
                segment_index_in_cycle = step_in_cycle // self.segment_duration # 0 for up, 1 for down (if mirror)

                if self.mirror and segment_index_in_cycle == 1:
                    # Down segment (only if mirror=True)
                    # We want the weight to go from 1.0 down to baseline.
                    # The annealer gives [baseline, 1] over 0..N-1 steps.
                    # To get [1, baseline], we map steps 0..N-1 -> N-1..0 for the annealer.
                    annealer_step_for_segment = self.segment_duration - 1 - step_in_segment # Maps 0..N-1 -> N-1..0
                else:
                    # Up segment (always the first segment, or the only segment if not mirror)
                    # We want the weight to go from baseline to 1.0.
                    annealer_step_for_segment = step_in_segment # Maps 0..N-1 -> 0..N-1

                # Get the base weight from the internal annealer [baseline, 1.0] (since annealer.reverse is False)
                base_weight = self.annealer.get_weight_at(annealer_step_for_segment)

                # Apply BetaAnnealer's reverse_segment flag AFTER getting the base weight [baseline, 1.0] range
                # If base_weight is 'w' in range [b, 1], its progress is (w - b) / (1 - b).
                # The reversed value is b + (1 - b) * (1 - progress).
                if self.reverse_segment and 1.0 - self.baseline > 1e-6:
                     progress_in_range = (base_weight - self.baseline) / (1.0 - self.baseline)
                     reversed_progress = 1.0 - progress_in_range
                     beta_weight = self.baseline + (1.0 - self.baseline) * reversed_progress
                else:
                     beta_weight = base_weight # Use weight from Annealer directly [baseline, 1.0] (or [1.0, baseline] if annealer.reverse was True, but we set it False)


                beta_value = self.max_beta * beta_weight # Scale the weight [baseline, 1] or [1, baseline] by max_beta


            elif self.cycle_mode == 'fu_et_al':
                # Fu et al. mode (ramp-up-hold)
                ramp_duration = self.segment_duration # Duration of the ramp segment (calculated in init)

                if step_in_cycle < ramp_duration:
                    # Ramp-up phase
                    # The internal annealer is configured for the ramp duration and baseline.
                    # It produces a weight [baseline, 1.0] over the ramp duration steps.
                    beta_weight = self.annealer.get_weight_at(step_in_cycle)
                    # Note: The reverse_segment flag of BetaAnnealer is typically NOT applied here,
                    # as Fu et al. schedule is usually a ramp *up*. If ramp down is needed,
                    # the internal Annealer could be initialized with reverse=True or another parameter added.
                    # Assuming ramp up [baseline*max_beta, max_beta].
                else:
                    # Hold phase - weight is fixed at 1.0
                    beta_weight = 1.0 # Fixed at maximum weight [1.0, 1.0]

                # Apply max_beta scaling to the weight [baseline, 1.0] or [1.0, 1.0]
                beta_value = self.max_beta * beta_weight


            return beta_value

        else:
            # Cooldown phase (or finished)
            return self.cooldown_value # Remains at cooldown_value

    def step(self):
        """
        Increments the total step counter.
        """
        # step is calculated based on the current phase and step in __call__.
        if self.current_step < self.total_duration_steps:
             self.current_step += 1

    def state_dict(self):
        """Returns the state of the beta annealer as a dict."""
        return {
            'current_step': self.current_step,
            # Internal annealer state is not needed as its calculation is based on current_step of BetaAnnealer
        }

    def load_state_dict(self, state_dict):
        """Loads the beta annealer state from a dict."""
        self.current_step = state_dict['current_step']

    def get_current_beta(self):
        """ Alias for __call__ for clarity. """
        return self.__call__()

    def get_total_duration(self):
        """ Returns the total number of steps the annealer will run through all phases. """
        return self.total_duration_steps

    def get_cycle_length(self):
        """ Returns the duration of a single annealing cycle (ramp + hold or up/down). """
        return self.cycle_length

    def get_annealing_duration_per_cycle(self):
         """ Returns the duration of the annealing part within a single cycle. """
         return self.cycle_length # This is the definition of cycle_length now

    def get_ramp_duration(self):
         """ Returns the duration of the ramp phase within a cycle. """
         return self.segment_duration # segment_duration is the duration of the ramp part for both modes


    def get_current_phase(self):
        """ Returns the current phase name ('warmup', 'annealing', 'cooldown', 'finished'). """
        if self.current_step < self.warmup_end:
            return 'warmup'
        elif self.current_step < self.annealing_end:
            return 'annealing'
        elif self.current_step < self.total_duration_steps:
            return 'cooldown'
        else:
            return 'finished' # Add a finished state after cooldown


    def get_step_in_phase(self):
        """ Returns the current step counter within the current phase. """
        if self.current_step < self.warmup_end:
            return self.current_step
        elif self.current_step < self.annealing_end:
            return self.current_step - self.warmup_end
        elif self.current_step < self.total_duration_steps:
            return self.current_step - self.annealing_end
        else:
            return self.current_step - self.total_duration_steps # Steps past the end



if __name__ == "__main__":
    import torch
    import numpy as np
    # Test the Annealer class
    print("--- Testing basic Annealer (Cyclical Cosine) ---")
    annealer = Annealer(total_steps=100, shape='cosine', baseline=0.1, cyclical=True)
    weights = [annealer.get_weight() for _ in range(150)]
    print(weights[:110]) # Print first cycle and a bit of the second
    for i in range(150):
        annealer.step()

    # Test the __call__ method
    print("\n--- Testing Annealer __call__ ---")
    annealer = Annealer(total_steps=10, shape='linear')
    kld_value = torch.tensor([5.0])
    for i in range(12):
        print(f"Step {i}: KLD before: {kld_value.item():.4f}, Annealed KLD: {annealer(kld_value).item():.4f}, Weight: {annealer.get_weight():.4f}")
        annealer.step()


    # Test the BetaAnnealer class with reverse=True (segment goes down)
    print("\n--- Testing BetaAnnealer (Reverse Segment, No Mirror/Warmup/Cooldown) ---")
    beta_annealer = BetaAnnealer(total_steps=10, max_beta=4.0, shape='linear', baseline=0.0, reverse=True)
    betas = [beta_annealer() for _ in range(12)]
    print(betas)
    for i in range(12): beta_annealer.step()


    # Test the BetaAnnealer class with reverse=False (segment goes up)
    print("\n--- Testing BetaAnnealer (Up Segment, No Mirror/Warmup/Cooldown) ---")
    beta_annealer = BetaAnnealer(total_steps=10, max_beta=4.0, shape='linear', baseline=0.0, reverse=False)
    betas = [beta_annealer() for _ in range(12)]
    print(betas)
    for i in range(12): beta_annealer.step()

    # Test BetaAnnealer with Warmup and Cooldown
    print("\n--- Testing BetaAnnealer (Warmup + Up + Cooldown) ---")
    beta_annealer = BetaAnnealer(total_steps=10, max_beta=5.0, shape='linear',
                                 warmup_steps=5, warmup_value=0.1,
                                 cooldown_steps=5, cooldown_value=0.5,
                                 baseline=0.0) # Baseline doesn't affect beta if max_beta is used directly?
                                                # Correction: Baseline affects the *shape* weight [baseline, 1]
    betas = [beta_annealer() for _ in range(beta_annealer.get_total_duration() + 2)]
    print(f"Total Duration: {beta_annealer.get_total_duration()}")
    print(betas)
    for i in range(beta_annealer.get_total_duration() + 2): beta_annealer.step()


    # Test BetaAnnealer with mirror and multiple cycles
    print("\n--- Testing BetaAnnealer (Warmup + Mirror Cycles + Cooldown) ---")
    beta_annealer = BetaAnnealer(total_steps=10, max_beta=4.0, shape='cosine', baseline=0.1,
                                 warmup_steps=5, warmup_value=0.05,
                                 mirror=True, num_cycles=3,
                                 cooldown_steps=5, cooldown_value=0.2)

    duration = beta_annealer.get_total_duration()
    print(f"Total Duration: {duration}")
    betas = [beta_annealer() for _ in range(duration + 2)]
    print(betas)
    for i in range(duration + 2): beta_annealer.step()

    # Test state_dict
    print("\n--- Testing State Dict ---")
    beta_annealer = BetaAnnealer(total_steps=10, max_beta=4.0, shape='linear',
                                 warmup_steps=5, warmup_value=0.1,
                                 cooldown_steps=5, cooldown_value=0.5)
    for _ in range(10):
        beta_annealer.step()
    state = beta_annealer.state_dict()
    print(f"State after 10 steps: {state}")

    new_beta_annealer = BetaAnnealer(total_steps=10, max_beta=4.0, shape='linear',
                                     warmup_steps=5, warmup_value=0.1,
                                     cooldown_steps=5, cooldown_value=0.5)
    new_beta_annealer.load_state_dict(state)

    print(f"Value after loading state: {new_beta_annealer()}")
    new_beta_annealer.step()
    print(f"Value after one step post-load: {new_beta_annealer()}")