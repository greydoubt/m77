from jinja2 import Environment, select_autoescape

env = Environment(autoescape=select_autoescape(
    enabled_extensions=('html', 'xml'),
    disabled_extensions=('txt',),
    default_for_string=True,
    default=True,
))
