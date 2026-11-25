"""Clinic no-show prediction: synthetic data, XGBoost model and Ray Serve service.

The package is split so that the "pure" parts (data generation, feature
engineering, risk thresholds, model wrapper, request handling) can be imported
and tested without starting Ray. Only :mod:`noshow.serve_app` depends on Ray
Serve, and it is a thin layer on top of :mod:`noshow.service`.
"""

__version__ = "0.1.0"
