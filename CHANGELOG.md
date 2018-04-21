# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `copy` for the core tensor structures
- Added function that describes an instance of `Tensor` class
- Added string descriptions for the modes of `Tensor`. Modes can also
be renamed



### Changed

- For all tensor representation all their data values can (should) only be accessed through corresponding properties.
- Objects of `Tensor` class can only be created from numpy arrays


### Removed



## [0.1.1]

### Added

- Core operations of tensor algebra
- Classes for different tensor representations (`Tensor`, `TensorCPD`, `TensorTKD`, `TensorTT`)
- Functions for computing special types of tensors (`super_diag_tensor`, `residual_tensor`)
- Implementation of the most fundamental tensor decompositions (`CPD`, `HOSVD`, `HOOI`,`TTSVD`)
- Several methods for computing metrics of tensor decompositions
- Functions for estimating optimal Kryskal rank and computing multi-linear rank of a `Tensor`
