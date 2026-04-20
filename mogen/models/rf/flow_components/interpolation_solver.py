import torch
import torch.nn as nn
import sympy
from typing import Callable

from ..utils import match_dim_with_data


class AffineInterpSolver:
    r"""Symbolic solver for affine interpolation equations.

    This class provides a symbolic solver for the affine interpolation equations:

        x_t = a_t * x_1 + b_t * x_0,
        dot_x_t = dot_a_t * x_1 + dot_b_t * x_0.

    Given at least two known variables among `x_0, x_1, x_t, dot_x_t`, and the rest unknown,
    the solver computes the unknowns. The method precomputes symbolic solutions for all pairs
    of unknown variables and stores them as lambdified functions for efficient numerical computation.
    """

    def __init__(self):
        r"""Initialize the `AffineInterpSolver` class.

        This method sets up the symbolic equations for affine interpolation and precomputes symbolic solvers
        for all pairs of unknown variables among `x_0, x_1, x_t, dot_x_t`. The equations are:

            x_t = a_t * x_1 + b_t * x_0,
            dot_x_t = dot_a_t * x_1 + dot_b_t * x_0.

        By solving these equations symbolically for each pair of unknown variables, the method creates lambdified
        functions that can be used for efficient numerical computations during runtime.
        """
        # Define symbols
        x_0, x_1, x_t, dot_x_t = sympy.symbols("x_0 x_1 x_t dot_x_t")
        a_t, b_t, dot_a_t, dot_b_t = sympy.symbols("a_t b_t dot_a_t dot_b_t")

        # Equations
        eq1 = sympy.Eq(x_t, a_t * x_1 + b_t * x_0)
        eq2 = sympy.Eq(dot_x_t, dot_a_t * x_1 + dot_b_t * x_0)

        # Variables to solve for
        variables = [x_0, x_1, x_t, dot_x_t]
        self.symbolic_solvers = {}

        # Create symbolic solvers for all pairs of unknown variables
        for i in range(len(variables)):
            for j in range(i + 1, len(variables)):
                unknown1, unknown2 = variables[i], variables[j]
                # print(f"Solving for {unknown1} and {unknown2}")
                # Solve equations
                solutions = sympy.solve([eq1, eq2], (unknown1, unknown2), dict=True)
                if solutions:
                    solution = solutions[0]
                    # Create lambdified functions
                    expr1 = solution[unknown1]
                    expr2 = solution[unknown2]
                    func = sympy.lambdify(
                        [x_0, x_1, x_t, dot_x_t, a_t, b_t, dot_a_t, dot_b_t],
                        [expr1, expr2],
                        modules="numpy",
                    )
                    # Store solver function
                    var_names = (str(unknown1), str(unknown2))
                    self.symbolic_solvers[var_names] = func

    def solve(
        self,
        results,
    ):
        r"""Solve for unknown variables in the affine interpolation equations.

        This method computes the unknown variables among `x_0, x_1, x_t, dot_x_t` given the known variables
        in the `results` object. It uses the precomputed symbolic solvers to find the solutions efficiently.

        Args:
            results (`Any`): An object (e.g., a dataclass or any object with attributes) **containing the following attributes**:
            - `x_0` (`torch.Tensor` or `None`): Samples from the source distribution.
            - `x_1` (`torch.Tensor` or `None`): Samples from the target distribution.
            - `x_t` (`torch.Tensor` or `None`): Interpolated samples at time `t`.
            - `dot_x_t` (`torch.Tensor` or `None`): The time derivative of `x_t` at time `t`.
            - `a_t`, `b_t`, `dot_a_t`, `dot_b_t` (`torch.Tensor`): Interpolation coefficients and their derivatives.

            Known variables should have their values assigned; unknown variables should be set to `None`.

        Returns:
            `Any`: The input `results` object with the unknown variables computed and assigned.

        Notes:
            - If only one variable among `x_0, x_1, x_t, dot_x_t` is unknown, the method selects an additional
              known variable to form a pair for solving.
            - The method assumes that at least two variables among `x_0, x_1, x_t, dot_x_t` are known.
            - The variables `a_t`, `b_t`, `dot_a_t`, and `dot_b_t` must be provided in `results`.

        Example:
            ```python
            >>> solver = AffineInterpSolver()
            >>> class Results:
            ...     x_0 = None
            ...     x_1 = torch.tensor([...])
            ...     x_t = torch.tensor([...])
            ...     dot_x_t = torch.tensor([...])
            ...     a_t = torch.tensor([...])
            ...     b_t = torch.tensor([...])
            ...     dot_a_t = torch.tensor([...])
            ...     dot_b_t = torch.tensor([...])
            >>> results = Results()
            >>> solver.solve(results)
            >>> print(results.x_0)  # Now x_0 is computed and assigned in `results`.
            ```
        """
        known_vars = {
            k: getattr(results, k)
            for k in ["x_0", "x_1", "x_t", "dot_x_t"]
            if getattr(results, k) is not None
        }
        unknown_vars = {
            k: getattr(results, k)
            for k in ["x_0", "x_1", "x_t", "dot_x_t"]
            if getattr(results, k) is None
        }
        unknown_keys = tuple(unknown_vars.keys())

        if len(unknown_keys) > 2:
            raise ValueError(
                "At most two variables among (x_0, x_1, x_t, dot_x_t) can be unknown."
            )
        elif len(unknown_keys) == 0:
            return results
        elif len(unknown_keys) == 1:
            # Select one known variable to make up the pair
            for var in ["x_0", "x_1", "x_t", "dot_x_t"]:
                if var in known_vars:
                    unknown_keys.append(var)
                    break

        func = self.symbolic_solvers.get(unknown_keys)

        # Prepare arguments in the order [x_0, x_1, x_t, dot_x_t, a_t, b_t, dot_a_t, dot_b_t]
        args = []
        for var in ["x_0", "x_1", "x_t", "dot_x_t", "a_t", "b_t", "dot_a_t", "dot_b_t"]:
            value = getattr(results, var, None)
            if value is None:
                value = 0  # Placeholder for unknowns
            args.append(value)

        # Compute the unknown variables
        solved_values = func(*args)
        # Assign the solved values back to results
        setattr(results, unknown_keys[0], solved_values[0])
        setattr(results, unknown_keys[1], solved_values[1])

        return results


class AffineInterp(nn.Module):
    r"""Affine Interpolation Module for Rectified Flow Models.

    This class implements affine interpolation between samples `x_0` from source distribution `pi_0` and
    samples `x_1` from target distribution `pi_1` over a time interval `t` in `[0, 1]`.

    The interpolation is defined using time-dependent coefficients `alpha(t)` and `beta(t)`:

        x_t = alpha(t) * x_1 + beta(t) * x_0,
        dot_x_t = dot_alpha(t) * x_1 + dot_beta(t) * x_0,

    where `x_t` is the interpolated state at time `t`, and `dot_x_t` is its time derivative.

    The module supports several predefined interpolation schemes:

    - **Straight Line Interpolation** (`"straight"` or `"lerp"`):

        alpha(t) = t,  beta(t) = 1 - t,
        dot_alpha(t) = 1, dot_beta(t) = -1.

    - **Spherical Interpolation** (`"spherical"` or `"slerp"`):

        alpha(t) = sin(pi / 2 * t), beta(t) = cos(pi / 2  * t),
        dot_alpha(t) = pi / 2 * cos(pi / 2 * t), dot_beta(t) = -pi / 2 * sin(pi / 2 * t).

    - **DDIM/DDPM Interpolation** (`"ddim"` or `"ddpm"`):

        alpha(t) = exp(-a * (1 - t) ** 2 / 4.0 - b * (1 - t) / 2.0),
        beta(t) = sqrt(1 - alpha(t) ** 2),
        a = 19.9 and b = 0.1.

    Attributes:
        name (`str`): Name of the interpolation scheme.
        alpha (`Callable`): Function defining `alpha(t)`.
        beta (`Callable`): Function defining `beta(t)`.
        dot_alpha (`Callable` or `None`): Function defining the time derivative `dot_alpha(t)`.
        dot_beta (`Callable` or `None`): Function defining the time derivative `dot_beta(t)`.
        solver (`AffineInterpSolver`): Symbolic solver for the affine interpolation equations.
        a_t (`torch.Tensor` or `None`): Cached value of `a(t)` after computation.
        b_t (`torch.Tensor` or `None`): Cached value of `b(t)` after computation.
        dot_a_t (`torch.Tensor` or `None`): Cached value of `dot_a(t)` after computation.
        dot_b_t (`torch.Tensor` or `None`): Cached value of `dot_b(t)` after computation.
    """

    def __init__(
        self,
        name: str = "straight",
        alpha: Callable | None = None,
        beta: Callable | None = None,
        dot_alpha: Callable | None = None,
        dot_beta: Callable | None = None,
    ):

        super().__init__()

        if name.lower() in ["straight", "lerp"]:
            # Special case for "straight" interpolation
            alpha = lambda t: t
            beta = lambda t: 1 - t
            dot_alpha = lambda t: torch.ones_like(t)
            dot_beta = lambda t: -torch.ones_like(t)
            name = "straight"
        elif name.lower() in ["slerp", "spherical"]:
            # Special case of "spherical" interpolation
            alpha = lambda t: torch.sin(t * torch.pi / 2.0)
            beta = lambda t: torch.cos(t * torch.pi / 2.0)
            dot_alpha = lambda t: torch.cos(t * torch.pi / 2.0) * torch.pi / 2.0
            dot_beta = lambda t: -torch.sin(t * torch.pi / 2.0) * torch.pi / 2.0
            name = "spherical"
        elif name.lower() in ["ddim", "ddpm"]:
            # DDIM/DDPM scheme; see Eq 7 in https://arxiv.org/pdf/2209.03003
            # Note in VP-ODE, dot_beta_t will explode at t = 1
            a = 19.9
            b = 0.1
            alpha = lambda t: torch.exp(-a * (1 - t) ** 2 / 4.0 - b * (1 - t) / 2.0)
            beta = lambda t: torch.sqrt(1 - alpha(t) ** 2)
            name = "DDIM"
        elif isinstance(name, str):
            raise ValueError(f"Unknown interpolation scheme: {name}. Provide custom interpolation functions for alpha and beta.")
        elif (
            alpha is None
            or beta is None
        ):
            # Custom interpolation functions
            raise NotImplementedError(
                "Custom interpolation functions must be provided for alpha and beta."
            )

        self.name = name
        self.alpha = lambda t: alpha(self.ensure_tensor(t))
        self.beta = lambda t: beta(self.ensure_tensor(t))
        if dot_alpha is not None:
            self.dot_alpha = lambda t: dot_alpha(self.ensure_tensor(t))
        else:
            self.dot_alpha = None
        if dot_beta is not None:
            self.dot_beta = lambda t: dot_beta(self.ensure_tensor(t))
        else:
            self.dot_beta = None

        self.solver = AffineInterpSolver()
        self.a_t = None
        self.b_t = None
        self.dot_a_t = None
        self.dot_b_t = None
        self.x_0 = None
        self.x_1 = None
        self.x_t = None
        self.dot_x_t = None

    @staticmethod
    def ensure_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.tensor(x)

    @staticmethod
    def value_and_grad(f, input_tensor, detach=True):
        r"""Compute the value and gradient of a function with respect to its input tensor.

        This method computes both the function value `f(x)` and its gradient `\nabla_x f(x)` at a given input tensor `x`.

        Args:
            f (`Callable`): The function `f` to compute.
            input_tensor (`torch.Tensor`): The input tensor.
            detach (`bool`, optional, defaults to `True`): Whether to detach the computed value and gradient from the computation graph.

        Returns:
            value_and_grad (Tuple[`torch.Tensor`, `torch.Tensor`]):
                `value`: The function value `f(x)`.
                `grad`: The gradient `\nabla_x f(x)`.

        Example:
            ```python
            >>> def func(x):
            ...     return x ** 2
            >>> x = torch.tensor(3.0, requires_grad=True)
            >>> value, grad = AffineInterp.value_and_grad(func, x)
            >>> value
            tensor(9.)
            >>> grad
            tensor(6.)
            ```
        """
        x = input_tensor.clone().detach().requires_grad_(True)
        with torch.enable_grad():
            value = f(x)
            (grad,) = torch.autograd.grad(value.sum(), x, create_graph=not detach)
        if detach:
            value = value.detach()
            grad = grad.detach()
        return value, grad

    def get_coeffs(self, t, detach=True):
        r"""Compute the interpolation coefficients `a_t`, `b_t`, and their derivatives `dot_a_t`, `dot_b_t` at time `t`.

        Args:
            t (`torch.Tensor`): Time tensor `t` at which to compute the coefficients.
            detach (`bool`, defaults to `True`): Whether to detach the computed values from the computation graph.

        Returns:
            coeff (Tuple[`torch.Tensor`, `torch.Tensor`, `torch.Tensor`, `torch.Tensor`]):
                `(a_t, b_t, dot_a_t, dot_b_t)`: The interpolation coefficients and their derivatives at time `t`.

        Notes:
            - If `dot_alpha` or `dot_beta` are not provided, their values are computed using automatic differentiation.
            - The computed coefficients are cached in the instance attributes `a_t`, `b_t`, `dot_a_t`, and `dot_b_t`.
        """
        if self.dot_alpha is None:
            a_t, dot_a_t = self.value_and_grad(self.alpha, t, detach=detach)
        else:
            a_t = self.alpha(t)
            dot_a_t = self.dot_alpha(t)
        if self.dot_beta is None:
            b_t, dot_b_t = self.value_and_grad(self.beta, t, detach=detach)
        else:
            b_t = self.beta(t)
            dot_b_t = self.dot_beta(t)
        self.a_t = a_t
        self.b_t = b_t
        self.dot_a_t = dot_a_t
        self.dot_b_t = dot_b_t
        return a_t, b_t, dot_a_t, dot_b_t

    def forward(self, x_0, x_1, t, detach=True):
        r"""Compute the interpolated `X_t` and its time derivative `dotX_t`.

        Args:
            x_0 (`torch.Tensor`): Samples from source distribution, shape `(B, D_1, D_2, ..., D_n)`.
            x_1 (`torch.Tensor`): Samples from target distribution, same shape as `x_0`.
            t (`torch.Tensor`): Time tensor `t`
            detach (`bool`, defaults to `True`): Whether to detach computed coefficients from the computation graph.

        Returns:
            interpolation (Tuple[`torch.Tensor`, `torch.Tensor`]):
                `x_t`: Interpolated state at time `t`.
                `dot_x_t`: Time derivative of the interpolated state at time `t`.
        """
        t = match_dim_with_data(t, x_1.shape, device=x_1.device, dtype=x_1.dtype)
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach=detach)
        x_t = a_t * x_1 + b_t * x_0
        dot_x_t = dot_a_t * x_1 + dot_b_t * x_0
        return x_t, dot_x_t

    def solve(self, t=None, x_0=None, x_1=None, x_t=None, dot_x_t=None, detach=True):
        r"""Solve for unknown variables in the affine interpolation equations.

        This method solves the equations:

            x_t = a_t * x_1 + b_t * x_0,
            dot_x_t = dot_a_t * x_1 + dot_b_t * x_0.

        Given any two of known variables among `x_0`, `x_1`, `x_t`, and `dot_x_t`, this method computes the unknown variables using the `AffineInterpSolver`.
        Must provide at least two known variables among `x_0`, `x_1`, `x_t`, and `dot_x_t`.

        Args:
            t (`torch.Tensor` or `None`):
                Time tensor `t`. Must be provided.
            x_0 (`torch.Tensor` or `None`, optional):
                Samples from the source distribution `pi_0`.
            x_1 (`torch.Tensor` or `None`, optional):
                Samples from the target distribution `pi_1`.
            x_t (`torch.Tensor` or `None`, optional):
                Interpolated samples at time `t`.
            dot_x_t (`torch.Tensor` or `None`, optional):
                Time derivative of the interpolated samples at time `t`.

        Returns:
            `AffineInterp`:
                The instance itself with the computed variables assigned to `x_0`, `x_1`, `x_t`, or `dot_x_t`.

        Raises:
            `ValueError`:
                - If `t` is not provided.
                - If less than two variables among `x_0`, `x_1`, `x_t`, `dot_x_t` are provided.

        Example:
            ```python
            >>> interp = AffineInterp(name='straight')
            >>> t = torch.tensor([0.5])
            >>> x_t = torch.tensor([[0.5]])
            >>> dot_x_t = torch.tensor([[1.0]])
            >>> interp.solve(t=t, x_t=x_t, dot_x_t=dot_x_t)
            >>> print(interp.x_0)  # Computed initial state x_0
            tensor([[0.]])
            >>> print(interp.x_1)  # Computed final state x_1
            tensor([[1.]])
            ```
        """
        if t is None:
            raise ValueError("t must be provided")

        self.x_0 = x_0
        self.x_1 = x_1
        self.x_t = x_t
        self.dot_x_t = dot_x_t

        non_none_values = [v for v in [x_0, x_1, x_t, dot_x_t] if v is not None]
        if len(non_none_values) < 2:
            raise ValueError("At least two of x_0, x_1, x_t, dot_x_t must not be None")

        x_not_none = non_none_values[0]
        t = match_dim_with_data(
            t, x_not_none.shape, device=x_not_none.device, dtype=x_not_none.dtype
        )
        a_t, b_t, dot_a_t, dot_b_t = self.get_coeffs(t, detach)

        return self.solver.solve(self)
