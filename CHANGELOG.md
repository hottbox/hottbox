# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [HOTTBOX v0.1.3 Unreleased]

### Package status on branch develop
[![Travis status](https://img.shields.io/travis/hottbox/hottbox/develop.svg?label=TravisCI)](https://travis-ci.org/hottbox/hottbox/)
[![Appveyor status](https://ci.appveyor.com/api/projects/status/2ct6ku31v351s3d3/branch/develop?svg=true)](https://ci.appveyor.com/project/IlyaKisil/hottbox-6jq6a/branch/develop)
[![Coveralls status](https://img.shields.io/coveralls/github/hottbox/hottbox/develop.svg)](https://coveralls.io/github/hottbox/hottbox)

### Added

- [ ] Quick construction of generic objects of `Tensor` class
- [ ] Direct summation of `Tensor` objects
- [ ] Method for resetting mode names
- [ ] Mode names for `TensorCPD` and `TensorTKD` classes
- [ ] Describe functions for `TensorCPD`, `TensorTKD` and `TensorTT`
- [ ] Option for sorting vectors for the `CPD`

### Changed

- [ ] The values of `_mode_names` dictionary are in form of list with length two. 
      The first element of such list specifies name for corresponding mode, whereas the second
      element is a place holder for index names 

### Removed

- [ ] Parameter `ft_shape` from the `TensorTT` constructor
- [ ] Parameter `mode_description` from constructors for all tensor decomposition algorithms


### Fixed

-



## HOTTBOX v0.1.2

### Added

- `copy` method for the core tensor structures
- `describe` method that describes an instance of `Tensor` class
- Mode descriptions for the modes of `Tensor` through the use of OrderedDict.
 Modes can also be renamed
- Input validation for constructors for `Tensor`, `TensorCPD`, `TensorTKD`, `TensorTT`
- Input validation for input data for `decompose` method for all tensor decomposition algorithms
- Setup CI using Travis, AppVeyor and Coveralls
- Unit tests using pytest for all available modules


### Changed

- Objects of `Tensor`, `TensorCPD`, `TensorTKD`, `TensorTT` classes can only be created from numpy arrays
- For all tensor representation all their data values can (should) only be accessed through corresponding properties.
- The original shape of the tensor can be defined during object creation of `Tensor` class
- `super_diag_tensor` requires to pass a shape of desired tensor instead of its order


### Fixed

- `reconstruct` was changing the original core so it was not possible to call it several times in a row
- Incorrect size of a produced factor matrix when its computation is skipped in `decompose` for `HOSVD` and `HOOI` classes 



## HOTTBOX v0.1.1

### Added

- Core operations of tensor algebra
- Classes for different tensor representations (`Tensor`, `TensorCPD`, `TensorTKD`, `TensorTT`)
- Functions for computing special types of tensors (`super_diag_tensor`, `residual_tensor`)
- Implementation of the most fundamental tensor decompositions (`CPD`, `HOSVD`, `HOOI`,`TTSVD`)
- Several methods for computing metrics of tensor decompositions
- Functions for estimating optimal Kryskal rank and computing multi-linear rank of a `Tensor`
