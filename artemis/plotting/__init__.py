
__author__ = 'peter'

from artemis.config import get_artemis_config
config = get_artemis_config()
if config.get('plotting', 'backend') == 'matplotlib-web':
    import matplotlib
    matplotlib.use('agg')
    from artemis.plotting.plotting_server import setup_web_plotting
    update_period = float(config.get('plotting', 'update_period'))
    setup_web_plotting(update_period=update_period)
