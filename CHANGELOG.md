# ChangeLog

## [Unversioned]

## [0.2.0] - 2024-01-24
- refactored code to factor out classifiers to aircraft_classifiers_jme45
- aircraft_classifiers_jme45 is no longer available on pypi, so installed from github

### [0.1.0] - 2024-01-08
- Converted code to use ml_utils_jme45 as an installable package
- Code works for basic use.
- Implemented trainable_parts to classifier, since for effnet cannot set entire model to .train()