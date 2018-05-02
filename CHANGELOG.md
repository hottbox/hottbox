# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased: develop branch]
[![Travis status](https://img.shields.io/travis/hottbox/hottbox/develop.svg?label=TravisCI)](https://img.shields.io/travis/hottbox/hottbox/develop.svg)
[![Appveyor status](https://ci.appveyor.com/api/projects/status/2ct6ku31v351s3d3/branch/develop?svg=true)](https://ci.appveyor.com/project/IlyaKisil/hottbox-6jq6a/branch/develop)
[![Coveralls status](https://img.shields.io/coveralls/github/hottbox/hottbox/develop.svg)](https://img.shields.io/coveralls/github/hottbox/hottbox/develop.svg)



### Added

- Added `copy` for the core tensor structures
- Added function that describes an instance of `Tensor` class
- Added string descriptions for the modes of `Tensor` through the use of OrderedDict.
 Modes can also be renamed
- Setup CI using Travis, AppVeyor and Coveralls



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
