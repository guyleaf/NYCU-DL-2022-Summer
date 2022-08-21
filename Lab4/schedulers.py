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
        initial_epoch: int = 0,
    ):
        super().__init__()
        periods = epoch_size / number_of_cycles
        self._steps = (stop - start) / (periods * ratio)
        self._start = start
        self._stop = stop
        self._beta = initial_epoch * self._steps
        self._check_beta()

    def _check_beta(self):
        if self._beta > self._stop:
            self._beta = self._start

    def step(self):
        self._beta += self._steps
        self._check_beta()

    def get_beta(self) -> float:
        return self._beta


class TeacherForcingScheduler:
    _epoch: int
    _ratio: float
    _start_decay_epoch: int
    _decay_step: float
    _min_ratio: float

    def __init__(
        self,
        ratio: float = 0.5,
        start_decay_epoch: int = 10,
        decay_step: float = 0.1,
        min_ratio: float = 0,
        initial_epoch: int = 0,
    ) -> None:
        super().__init__()
        diff_epochs = initial_epoch - start_decay_epoch + 1
        if diff_epochs > 0:
            ratio -= decay_step * diff_epochs
            ratio = min(ratio, min_ratio)

        self._epoch = initial_epoch
        self._ratio = ratio
        self._start_decay_epoch = start_decay_epoch
        self._decay_step = decay_step
        self._min_ratio = min_ratio

    def step(self):
        self._epoch += 1
        if self._epoch >= self._start_decay_epoch:
            self._ratio = max(self._ratio - self._decay_step, self._min_ratio)

    def get_ratio(self):
        return self._ratio
