"""EAGLE OpenFlow controller and process supervisor."""

from pkgutil import extend_path

from pkg_resources import get_distribution

# Share this package with the installed Ryu library in the Python 3.7 environment.
__path__ = extend_path(__path__, __name__)
# The upstream command manager imports this attribute from the root package.
version = get_distribution("ryu").version
