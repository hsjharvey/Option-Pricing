import warnings


def barrier_warning(option_type: str, barrier_type: str, barrier_direction: str, barrier_price: float,
                    s: float) -> None:
    warnings.warn(
        f"Barrier condition has already been triggered: {option_type} "
        f"{barrier_type} {barrier_direction} spot price:{self.S0} vs. barrier price: {barrier_price}")
