# Logging Utilities

This module provides utility functions for logging operations within the MyoLab API.

- [logger](logger.py) implementes a comprehensive logger class with multiple backends (such as WandB) to sync the logs
- [grouped_dataset](grouped_dataset.py) is a custom logger to facilitates working with grouped data. Originally designed by Robohive project, it is quite general and can also be used for recording general grouped datasets across many projects. Refer to its [original documentation](https://github.com/vikashplus/robohive/tree/main/robohive/logger) for more details.
