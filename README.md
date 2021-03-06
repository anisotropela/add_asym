<!--
<link href="./styles.css" rel="stylesheet"></link>
<script id="MathJax-script" async src="./tex-mml-chtml.js"></script>
-->

# `add_asym`

## Description

`add_asym` is a Python code for adding numbers with asymmetric uncertainties. Basically just a numerical implementation of the statistical methods discussed in [Barlow (2003, arXiv:0306138)](https://ui.adsabs.harvard.edu/abs/2003physics...6138B), the code was developed as a part of the work leading to the paper [Lyman α-emitting galaxies in the epoch of reionization](https://arxiv.org/abs/1806.07392v1).

## Acknowledgments

If you use the code for scientific use, I'd be happy if you could cite [Laursen et al. (2019, A&A, 627, 84)](https://ui.adsabs.harvard.edu/abs/2019A%26A...627A..84L),
 where the code is described, tested, and applied.

## Documentation

I will write a better description in the bottom of this document <s>soon</s> at some point, but really all you need is the file `add_asym.py`, in which the docstring describes how — and in particular *why* — to use the code.

## Usage

To add the following numbers
	
![equation](http://latex.codecogs.com/gif.latex?5_{-2}^{+1}\\,\\,+\\,\\,3_{-3}^{+1}\\,\\,+\\,\\,4_{-3}^{+2})

<!--
5<sup style='position: relative; left: +.1em; top: -.3em;'>+1</sup><sub style='position: relative; left: -1.1em; top: .2em;'>–2</sub>+
&nbsp;
3<sup style='position: relative; left: +.1em; top: -.3em;'>+1</sup><sub style='position: relative; left: -1.1em; top: .2em;'>–3</sub>+
&nbsp;
4<sup style='position: relative; left: +.1em; top: -.3em;'>+2</sup><sub style='position: relative; left: -1.1em; top: .2em;'>–3</sub>,
-->

<!--
$$
\qquad 5_{-2}^{+1}  \,+\,  3_{-3}^{+1}  \,+\,  4_{-3}^{+2},
$$
-->
define arrays/lists containing the central values, lower errors, and upper errors, respectively, and call `add_asym` with these arrays, using either a linear transformation (`order=1`) or a quadratic transformation (`order=2`):

```python
>>> from add_asym import add_asym
>>> x0 = [5,3,4]                # Central values
>>> s1 = [2,3,3]                # Lower errors (do not include the minus)
>>> s2 = [1,1,2]                # Upper errors
>>> add_asym(x0,s1,s2,order=1)  # Add numbers (using a linear transformation)
array([10.92030132,  4.23748786,  2.94389109])
```

The output is the central value, the lower error, and the upper error, respectively. That is, the result would be written ![equation](http://latex.codecogs.com/gif.latex?10.9_{-4.2}^{+2.9}). Using `order=2` yields a slightly different result, namely ![equation](http://latex.codecogs.com/gif.latex?10.7_{-4.5}^{+3.2}). None of these can be said to be more correct than the other when the full PDFs are not known. But in general, both results will be a better approximation than the standard, but wrong, way of adding upper and lower errors in quadrature separately, i.e.

![equation](http://latex.codecogs.com/gif.latex?5_{-2}^{+1}\\,+\\,3_{-3}^{+1}\\,+\\,4_{-3}^{+2}\\,\\,\\,=\\,\\,\\,(5+3+4)_{-\\sqrt{2^2+3^2+3^2}}^{+\\sqrt{1^2+1^2+2^2}})

![equation](http://latex.codecogs.com/gif.latex?.\\hspace{3.25cm}\\simeq\\,\\,12_{-4.7}^{+2.4}.\\hspace{3cm}(\\mathrm{WRONG!}))

<!--$$
  \begin{array}{rcl}
   5_{-2}^{+1}  \,+\,  3_{-3}^{+1}  \,+\,  4_{-3}^{+2} & = & (5+3+4)_{-\sqrt{2^2+3^2+3^2}}^{+\sqrt{1^2+1^2+2^2}} \\
   & \simeq & 12_{-4.7}^{+2.4}. \hspace{3cm}(\mathrm{WRONG!})
  \end{array}
$$
-->

This result can be acheived setting `order=0`, but will output a warning:

```python
>>> add_asym(x0,s1,s2,order=0)
This is WRONG! Mark my words, it is WROOOOOONG!!!
array([12.        ,  4.69041576,  2.44948974])
```

<!--
## Motivation for using `add_asym`
-->