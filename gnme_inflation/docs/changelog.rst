*********
Changelog
*********

2.0.3 - 2025-08-08
******************

* New: Added support for symmetries in the target distributions. This allows to further simplify the optimization problems when the target distribution is known to have some symmetries (for instance, under relabeling of parties, or under relabeling of outcomes). See example of use in the `Advanced  <https://ecboghiu.github.io/inflation/_build/html/advanced.html>`_ page.

* New: Initial and experimental support for quantum theory based only on real numbers.

* Improved: The arguments for ``InflationSDP.build_columns`` can be now directly passed onto ``InflationSDP.generate_relaxation``. This means that one can now build directly SDPs for restricted generating sets just by calling, e.g., ``InflationSDP.generate_relaxation("local3", max_monomial_length=2)``.

* Improved: The computation of dual LPs is now handled automatically by MOSEK. This improves the reliability of the code (and reduces the maintenance effort). The analogous for SDPs will come soon.

* Improved: The error messages for unsupported scenarios (namely, structures where a parent and a child node do not share a latent common cause, and structures with intermediate latents that have observable parents) are now more informative.

* Fixed: The output status of LPs and SDPs is now the same.

2.0.2 - 2024-12-14
******************

* Compatibility fixes and improvements: the uses of SciPy's `coo_matrix` (to be deprecated) have been substituted by `coo_array`. Also, the library does not need `networkx` any more.

* Bug fixes in the generation of physical monomials in structures with intermediate latents.

* Memory and runtime improvements.

2.0.1 - 2024-10-21
******************

* Support for `NumPy 2 <https://numpy.org/devdocs/release/2.0.0-notes.html>`_.

* Small bugfixes.

* Memory and runtime improvements.

2.0.0 - 2024-06-01
******************

* Added support for linear programming relaxations of causal scenarios, via ``InflationLP``. This allows to run inflation hierarchies bounding the sets of classical and no-signaling correlations, per `J. Causal Inference 7(2), 2019 <https://doi.org/10.1515/jci-2017-0020>`_ (`arXiv:1609.00672 <https://arxiv.org/abs/1609.00672>`_)

* Added support for hybrid scenarios with sources of different nature, via the ``classical_sources`` argument to ``InflationProblem``. Currently supported: classical-no-signaling (via ``InflationLP``) and classical-quantum (via ``InflationSDP``).

* Added support for possibilistic-type feasibility problems (via ``supports_problem`` in ``InflationLP`` and ``InflationSDP``).

* Added initial support for structures with multiple layers of latent variables.

* Improved support for structures with visible-to-visible connections, using the notation of do-conditionals.

* Improved handling of certificates. This makes them easier to manipulate and evaluate.

* Revamped the description of monomials. This makes the codes faster and consume less memory.

1.0.0 - 2022-11-28
******************

* Initial release.
