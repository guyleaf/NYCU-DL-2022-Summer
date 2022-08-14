class KLAnnealingScheduler:
    _start: float
    _stop: float
    _steps: float
    _beta: float = 0

    def __init__(
        self,
        epoch_size: int,
        number_of_cycles: int = 4,
        ratio: float = 0.5,
        start: float = 0,
        stop: float = 1,
    ):
        super().__init__()
        periods = epoch_size / number_of_cycles
        self._steps = (stop - start) / (periods * ratio)
        self._start = start
        self._stop = stop

    def step(self):
        self._beta += self._steps
        if self._beta > self._stop:
            self._beta = self._start

    def get_beta(self) -> float:
        return self._beta
