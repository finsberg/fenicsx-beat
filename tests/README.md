# Analytic solutions and convergence tests

To verify our solvers we derive some analytic solutions and checks the numerical solution against the analytic. We also perform convergence tests in the spatial and temporal dimension and verify that we get the correct order of convergence.

## Spatial convergence tests
TBW

## Temporal convergence tests
TBW

## Analytic solution to the Monodomain model

We want to solve

\begin{align*}
\frac{\partial v}{\partial t} + I_{\mathrm{ion}}(v, s) - \nabla \cdot \left( M \nabla v \right) - I_{\mathrm{stim}} &= 0 \\
\frac{\partial s}{\partial t} &= f(s, v, t)
\end{align*}

for $(x, y) \in \Omega$,
with the boundary condition

$$
n \cdot \left( M \nabla v \right) = 0,
$$

for  $(x, y) \in \partial \Omega$. If we let $\Omega = [0, 1] \times [0, 1]$, $M = 1.0$, and let

$$
v(x, y, t) = \cos(2\pi x)\cos(2\pi y)\sin(t),
$$

then

$$
\frac{\partial v}{\partial t} = \cos(2\pi x)\cos(2\pi y)\cos(t),
$$

and


$$
\nabla v = -2\pi \sin(t) \begin{pmatrix} \sin(2\pi x)\cos(2\pi y)  \\  \cos(2\pi x)\sin(2\pi y) \end{pmatrix},
$$

and finally

$$
\nabla \cdot \nabla v = -8\pi^2 \cos(2\pi x)\cos(2\pi y)\sin(t)
$$

Therefore we see that

$$
n \cdot \left(\nabla v \right) = 0,
$$

for $x \in \partial \Omega$, and $n$ being the outward pointing unit normal. Since

$$
I_{\mathrm{stim}} = \frac{\partial v}{\partial t} + I_{\mathrm{ion}}(v, s) - \nabla \cdot \left( M \nabla v \right),
$$

we have

$$
I_{\mathrm{stim}} = \cos(2\pi x)\cos(2\pi y) (\cos(t) + 8\pi^2 \sin(t)) + I_{\mathrm{ion}}.
$$

Now we can let

$$
I_{\mathrm{ion}} = s,
$$

and

$$
\frac{\partial s}{\partial t} = v.
$$

Then

$$
s= -\cos(2\pi x)\cos(2\pi y)\cos(t),
$$

and finally

$$
I_{\mathrm{stim}} =  8\pi^2 \cos(2\pi x)\cos(2\pi y)\sin(t).
$$

We also have the following initial conditions

\begin{align*}
v(0) &= 0\\
s(0) &= -\cos(2\pi x)\cos(2\pi y)
\end{align*}

## Splitting scheme

1. Solve

    \begin{align*}
    \frac{\partial v}{\partial t} &= -I_{\mathrm{ion}}(v, s), \hspace{1cm} v(t_n) = v^n\\
    \frac{\partial s}{\partial t} &= f(s, v, t) \hspace{1cm} s(t_n) = s^n
    \end{align*}

    for $t_n < t \leq t_n + \theta \Delta t$. The solutions at $t_n + \theta \Delta t$ are denoted $s_{\theta}^n$ and  $v_{\theta}^n$.

2. Solve

    \begin{align*}
    \frac{\partial v}{\partial t} = \nabla \cdot \left( M \nabla v \right)  + I_{\mathrm{stim}}, \hspace{1cm} v(t_n) = v_{\theta}^n
    \end{align*}

    for $t_n < t \leq t_n + \Delta t$. The solution is denoted $v_{\theta}^{n+1}$

3. Solve

    \begin{align*}
    \frac{\partial v}{\partial t} &= -I_{\mathrm{ion}}(v, s), \hspace{1cm} v(t_n + \theta \Delta t) = v_{\theta}^{n+1}\\
    \frac{\partial s}{\partial t} &= f(s, v, t) \hspace{1cm} s(t_n + \theta \Delta t) = s_{\theta}^{n}
    \end{align*}

    for $t_n + \theta \Delta t < t \leq t_n + \Delta t$. The solutions at $t_n + \theta \Delta t$ are denoted $s^{n+1}$ and  $v^{n+1}$.
