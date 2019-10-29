import numpy as np
from scipy import integrate
from .clib.twiss_product import twiss_product, accumulated_array
from .matrix_method import MatrixMethod
from .utils import Signal

CONST_C = 299_792_458
TWO_PI = 2 * np.pi


class Twiss:
    def __init__(self, main_cell):
        self.main_cell = main_cell
        self.matrix_method = MatrixMethod(main_cell)

        self._twiss_array = None
        self._twiss_array_needs_update = True
        self.twiss_array_changed = Signal(self.matrix_method.matrix_array_changed)
        self.twiss_array_changed.register(self.on_twiss_array_changed)
        self._full_matrix = None
        self._accumulated_array = None

        self.stable = None
        self.stable_x = None
        self.stable_y = None

        self._tune_fractional_needs_update = True
        # TODO: only depends on full-matrix not on beta functions - change signals!
        self.tune_fractional_changed = Signal(self.twiss_array_changed)
        self.tune_fractional_changed.register(self.on_tune_fractional_changed)
        self._tune_x_fractional = None
        self._tune_y_fractional = None
        self._tune_x_fractional_hz = None
        self._tune_y_fractional_hz = None

        self._betatron_phase_needs_update = True
        self.betatron_phase_changed = Signal(self.twiss_array_changed)
        self.betatron_phase_changed.register(self.on_betatron_phase_changed)
        self._psi_x = None
        self._psi_y = None
        self._tune_x = None
        self._tune_y = None

    @property
    def twiss_array(self) -> np.ndarray:
        if self._twiss_array_needs_update:
            self.update_twiss_array()

        return self._twiss_array

    @property
    def full_matrix(self) -> np.ndarray:
        if self._twiss_array_needs_update:
            self.update_twiss_array()

        return self._full_matrix

    @property
    def accumulated_array(self) -> np.ndarray:
        if self._twiss_array_needs_update:
            self.update_twiss_array()

        return self._accumulated_array

    def update_twiss_array(self):
        transfer_matrices = self.matrix_method.matrix_array
        size = transfer_matrices.shape[0]
        if self._twiss_array is None or self._twiss_array.shape[0] != size:
            self._twiss_array = np.empty((8, size))
            self._accumulated_array = np.empty(transfer_matrices.shape)
        accumulated_array(transfer_matrices, self._accumulated_array)
        self._full_matrix = full_matrix = self._accumulated_array[-1]

        term_x = 2 - full_matrix[0, 0] ** 2 - 2 * full_matrix[0, 1] * full_matrix[1, 0] - full_matrix[1, 1] ** 2
        self.stable_x = term_x > 0

        term_y = 2 - full_matrix[2, 2] ** 2 - 2 * full_matrix[2, 3] * full_matrix[3, 2] - full_matrix[3, 3] ** 2
        self.stable_y = term_y > 0
        self.stable = self.stable_x and self.stable_y

        if not self.stable:
            pass
            # warnings.warn(f"Horizontal plane stability: {twiss.stable_x}\nVertical plane stability{twiss.stable_y}")
        else:
            beta_x0 = np.abs(2 * full_matrix[0, 1]) / np.sqrt(term_x)
            alpha_x0 = (full_matrix[0, 0] - full_matrix[1, 1]) / (2 * full_matrix[0, 1]) * beta_x0
            gamma_x0 = (1 + alpha_x0 ** 2) / beta_x0
            beta_y0 = np.abs(2 * full_matrix[2, 3]) / np.sqrt(term_y)
            alpha_y0 = (full_matrix[2, 2] - full_matrix[3, 3]) / (2 * full_matrix[2, 3]) * beta_y0
            gamma_y0 = (1 + alpha_y0 ** 2) / beta_y0
            eta_x_dds0 = (full_matrix[1, 0] * full_matrix[0, 5] + full_matrix[1, 5] * (1 - full_matrix[0, 0])) / (
                    2 - full_matrix[0, 0] - full_matrix[1, 1])
            eta_x0 = (full_matrix[0, 1] * eta_x_dds0 + full_matrix[0, 5]) / (1 - full_matrix[1, 1])

            initial_twiss_vec = np.array([beta_x0, beta_y0, alpha_x0, alpha_y0, gamma_x0, gamma_y0, eta_x0, eta_x_dds0])
            twiss_product(self._accumulated_array, initial_twiss_vec, self._twiss_array)

        self._twiss_array_needs_update = False

    def on_twiss_array_changed(self):
        self._twiss_array_needs_update = True

    @property
    def s(self) -> np.ndarray:
        return self.matrix_method.s

    @property
    def beta_x(self) -> np.ndarray:
        return self.twiss_array[0]

    @property
    def beta_y(self) -> np.ndarray:
        return self.twiss_array[1]

    @property
    def alpha_x(self) -> np.ndarray:
        return self.twiss_array[2]

    @property
    def alpha_y(self) -> np.ndarray:
        return self.twiss_array[3]

    @property
    def gamma_x(self) -> np.ndarray:
        return self.twiss_array[4]

    @property
    def gamma_y(self) -> np.ndarray:
        return self.twiss_array[5]

    @property
    def eta_x(self) -> np.ndarray:
        return self.twiss_array[6]

    @property
    def eta_x_dds(self) -> np.ndarray:
        return self.twiss_array[7]

    @property
    def psi_x(self) -> np.ndarray:
        if self._betatron_phase_needs_update:
            self.update_betatron_phase()
        return self._psi_x

    @property
    def psi_y(self) -> np.ndarray:
        if self._betatron_phase_needs_update:
            self.update_betatron_phase()
        return self._psi_y

    @property
    def tune_x(self) -> float:
        if self._betatron_phase_needs_update:
            self.update_betatron_phase()
        return self._tune_x

    @property
    def tune_y(self) -> float:
        if self._betatron_phase_needs_update:
            self.update_betatron_phase()
        return self._tune_y

    def update_betatron_phase(self):
        size = self.accumulated_array.shape[0]
        self._psi_x = np.empty(size)  # TODO: do not always allocate new!
        self._psi_y = np.empty(size)
        beta_x_inverse = 1 / self.beta_x
        beta_y_inverse = 1 / self.beta_y
        self._psi_x = integrate.cumtrapz(beta_x_inverse, self.s, initial=0)  # TODO: use faster integration!
        self._psi_y = integrate.cumtrapz(beta_y_inverse, self.s, initial=0)
        self._tune_x = self._psi_x[-1] / TWO_PI
        self._tune_y = self._psi_y[-1] / TWO_PI
        self._betatron_phase_needs_update = False

    def on_betatron_phase_changed(self):
        self._betatron_phase_needs_update = True

    @property
    def tune_x_fractional(self) -> float:
        if self._tune_fractional_needs_update:
            self.update_fractional_tune()

        return self._tune_x_fractional

    @property
    def tune_y_fractional(self) -> float:
        if self._tune_fractional_needs_update:
            self.update_fractional_tune()

        return self._tune_y_fractional

    @property
    def tune_x_fractional_hz(self) -> float:
        if self._tune_fractional_needs_update:
            self.update_fractional_tune()

        return self._tune_x_fractional_hz

    @property
    def tune_y_fractional_hz(self) -> float:
        if self._tune_fractional_needs_update:
            self.update_fractional_tune()

        return self._tune_y_fractional_hz

    def update_fractional_tune(self):
        full_matrix = self.full_matrix
        self._tune_x_fractional = np.arccos((full_matrix[0, 0] + full_matrix[1, 1]) / 2) / TWO_PI
        self._tune_y_fractional = np.arccos((full_matrix[2, 2] + full_matrix[3, 3]) / 2) / TWO_PI
        tmp = self.matrix_method.velocity / self.main_cell.length  # Hz
        self._tune_x_fractional_hz = self._tune_x_fractional * tmp
        self._tune_y_fractional_hz = self._tune_y_fractional * tmp
        self._tune_fractional_needs_update = False

    def on_tune_fractional_changed(self):
        self._tune_fractional_needs_update = True

    # TODO: save results
    def beta_x_int(self, steps):
        s_int = np.linspace(0, self.s[-1], steps)
        return np.interp(s_int, self.s, self.beta_x)

    def beta_y_int(self, steps):
        s_int = np.linspace(0, self.s[-1], steps)
        return np.interp(s_int, self.s, self.beta_x)