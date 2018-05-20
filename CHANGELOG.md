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

- [x] Class `Mode` for meta information for tensor representations (as a part of pandas and plotting integration)
- [x] Methods for (re)setting mode names and the corresponding indices for `Tensor`
- [x] Placeholder `_state` which tracks changes of the `Tensor`
- [x] Mode description (and the corresponding methods) for `TensorCPD`, `TensorTKD` and `TensorTT` classes 
      by analogy with the `Tensor` class
- [x] Parameter `keep_meta` to `decompose` methods for the cpd and tucker type decompositions.
      Base on its value, meta information of the modes of `tensor` to be decomposed can be extracted
      and assigned to the `TensorCPD` and `TensorTKD` respectively. 
- [x] Quick construction of generic objects of `Tensor` class
- [ ] Quick construction of generic objects of `TensorCPD`, `TensorTKD` and `TensorTT` classes
- [x] Direct summation and comparison of `Tensor` objects (redefined \_\_add\_\_, \_\_eq\_\_)
- [ ] Vectorisation method for a `Tensor` class 
- [ ] Describe functions for `TensorCPD`, `TensorTKD` and `TensorTT`
- [ ] Option for sorting vectors for the `CPD`
- [x] Restrictions on methods `fold`, `unfold` and `mode_n_product` of `Tensor`.
      Whether they can be called is determined by the current state of the `Tensor` object.
- [x] Tools to convert multi-index pandas dataframe into a `Tensor` and vise versa 


### Changed

- [x] Each mode of a `Tensor` there is characterised by a corresponding `Mode` object with meta information
- [x] Mode names for the `Tensor` constructor should be passed as list instead of OrderedDict.
      These names are used to create `Mode` objects which are stored in a  list `Tensor._modes`
- [x] Property `reconstruct` of `TensorCPD`, `TensorTKD` and `TensorTT` classes is now a method 
      (should have been in the first place). Also it take optional parameter `keep_mata` for extraction
      of meta information about modes
       

### Removed

- [ ] Parameter `ft_shape` from the `TensorTT` constructor
- [x] Parameter `mode_description` from constructors for all tensor decomposition algorithms
- [x] Attribute `_mode_names` from the `Tensor`


### Fixed

- [ ] Index change for during `mode_n_product`
- [ ] Fix copy methods for `TensorCPD` and `TensorTKD` due to new attributes



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
