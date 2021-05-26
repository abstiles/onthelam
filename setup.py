#!/usr/bin/env python

import pkg_resources
import setuptools

# This is the minimum version of setuptools required to parse the contents of
# setup.cfg. This line ensures an appropriate error is printed.
pkg_resources.require('setuptools>=39.2')

# Even if the setup is not configured in this file, we need an empty call to
# setup() in order to support editable installs.
setuptools.setup()
